import os
import sys
import json
import argparse
import cv2
import torch
import clip
from PIL import Image

DETIC_DIR = os.path.expanduser("~/Desktop/drspring/realrobot/Detic")
sys.path.insert(0, DETIC_DIR)
sys.path.insert(0, os.path.join(DETIC_DIR, "third_party/CenterNet2"))

from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

DETIC_VOCAB = ["cone", "traffic cone", "traffic cylinder", "traffic pole", "dustbin", "barrel"]

ASSET_TYPES = ["traffic_cone", "traffic_cylinder", "traffic_pole", "dustbin"]

DETIC_TO_ASSET = {
    "cone": "traffic_cone",
    "traffic cone": "traffic_cone",
    "traffic cylinder": "traffic_cylinder",
    "traffic pole": "traffic_pole",
    "dustbin": "dustbin",
    "barrel": "traffic_cylinder",  # barrels are usually traffic cylinders in road scenes
}

CLIP_PROMPTS = [
    "a photo of a traffic cone on the road",
    "a photo of a traffic barrel or traffic cylinder on the road",
    "a photo of a traffic pole or post on the road",
    "a photo of a dustbin or trash can on the road",
]

DETIC_CONFIG = os.path.join(DETIC_DIR,
    "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
DETIC_WEIGHTS = os.path.join(DETIC_DIR,
    "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")

OUTPUT_DIR = "output"
DETIC_CONF = 0.3
POLE_MIN_CONF = 0.5      
POLE_MIN_ASPECT = 3.0     
MAX_OBJECTS_PER_FRAME = 8 
MIN_CROP_PX = 15


# model loading
def build_detic(device):
    orig_cwd = os.getcwd()
    os.chdir(DETIC_DIR)

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(DETIC_CONFIG)
    cfg.MODEL.WEIGHTS = DETIC_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETIC_CONF
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)

    from detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    prompts = [f"a {c}" for c in DETIC_VOCAB]
    with torch.no_grad():
        tf = text_encoder(prompts).float()
        tf = tf / tf.norm(dim=-1, keepdim=True)
        tf = tf.permute(1, 0).contiguous()
    reset_cls_test(predictor.model, tf, len(DETIC_VOCAB))

    os.chdir(orig_cwd)
    return predictor


def load_clip_model(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize(CLIP_PROMPTS).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features


# classification
def clip_classify(crop_bgr, clip_model, clip_preprocess, clip_text, device):
    """CLIP zero-shot on crop, returns (asset_type, confidence)."""
    h, w = crop_bgr.shape[:2]
    if h < MIN_CROP_PX or w < MIN_CROP_PX:
        return None, 0.0
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    inp = clip_preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sims = (feat @ clip_text.T)[0]
        probs = torch.softmax(sims, dim=0)
    idx = probs.argmax().item()
    return ASSET_TYPES[idx], probs[idx].item()


def ensemble_classify(detic_cls, crop_bgr, clip_model, clip_preprocess, clip_text, device):
    """Detic proposes, CLIP verifies. If they agree → high confidence.
    If they disagree → trust CLIP (it sees the crop directly)."""
    detic_asset = DETIC_TO_ASSET.get(detic_cls, "traffic_cone")
    clip_asset, clip_conf = clip_classify(crop_bgr, clip_model, clip_preprocess, clip_text, device)

    if clip_asset is None:
        return detic_asset  # too small for CLIP, trust Detic

    if detic_asset == clip_asset:
        return detic_asset  # agreement
    else:
        return clip_asset   # CLIP overrides on crop


# helpers
def load_frame(scene, frame_idx):
    for ext in (".jpg", ".png"):
        path = os.path.join(OUTPUT_DIR, scene, "frames", f"frame_{frame_idx:05d}{ext}")
        if os.path.exists(path):
            return cv2.imread(path)
    return None


def iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    aa = (a[2]-a[0])*(a[3]-a[1])
    ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0


# main pipeline
def process_scene(scene, predictor, clip_model, clip_preprocess, clip_text, device, step=1):
    det_path = os.path.join(OUTPUT_DIR, scene, "detections.json")
    if not os.path.exists(det_path):
        print(f"    skip: no detections.json")
        return

    with open(det_path) as f:
        frames_data = json.load(f)

    # build frame_idx → entry index map
    fidx_to_ei = {entry["frame_idx"]: ei for ei, entry in enumerate(frames_data)}

    total_added = 0

    for ei, entry in enumerate(frames_data):
        # subsample: process only every Nth entry (matches render cadence)
        if step > 1 and ei % step != 0:
            continue
        fidx = entry["frame_idx"]
        frame = load_frame(scene, fidx)
        if frame is None:
            continue

        # run Detic on full frame
        outputs = predictor(frame)
        instances = outputs["instances"].to("cpu")

        existing_bboxes = [d["bbox"] for d in entry.get("detections", [])]

        # collect candidates sorted by confidence
        candidates = []
        for i in range(len(instances)):
            bbox = instances.pred_boxes.tensor[i].numpy().tolist()
            cls_idx = instances.pred_classes[i].item()
            score = instances.scores[i].item()
            detic_cls = DETIC_VOCAB[cls_idx]

            # filter noisy pole detections
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if detic_cls == "traffic pole":
                if score < POLE_MIN_CONF:
                    continue
                if bh / max(bw, 1) < POLE_MIN_ASPECT:
                    continue

            # skip if overlaps with existing detection (vehicle, ped, etc.)
            overlaps = any(iou(bbox, eb) > 0.5 for eb in existing_bboxes)
            if overlaps:
                continue

            candidates.append((score, bbox, detic_cls))

        # sort by confidence, cap per frame
        candidates.sort(key=lambda x: -x[0])
        candidates = candidates[:MAX_OBJECTS_PER_FRAME]

        for score, bbox, detic_cls in candidates:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

            final_type = ensemble_classify(
                detic_cls, crop, clip_model, clip_preprocess, clip_text, device
            )

            new_det = {
                "label": final_type,
                "bbox": bbox,
                "confidence": score,
                "camera": "front",
            }
            entry["detections"].append(new_det)
            existing_bboxes.append(bbox)
            total_added += 1

    with open(det_path, "w") as f:
        json.dump(frames_data, f)
    print(f"    {total_added} additional objects detected")


def main():
    parser = argparse.ArgumentParser(description="Additional object detection (Detic + CLIP)")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--step", type=int, default=1,
                        help="process every Nth detection entry (matches render cadence)")
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading Detic...")
    predictor = build_detic(device)

    print("Loading CLIP...")
    clip_model, clip_preprocess, clip_text = load_clip_model(torch.device(device))

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted([
            d for d in os.listdir(OUTPUT_DIR)
            if os.path.isdir(os.path.join(OUTPUT_DIR, d))
            and os.path.exists(os.path.join(OUTPUT_DIR, d, "detections.json"))
        ])

    print(f"Processing {len(scenes)} scenes...")
    for scene in scenes:
        print(f"  → {scene}")
        process_scene(scene, predictor, clip_model, clip_preprocess, clip_text, device, step=args.step)

    print("Done.")


if __name__ == "__main__":
    main()
