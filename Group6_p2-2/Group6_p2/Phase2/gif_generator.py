import imageio, os, glob, argparse

_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--render_dir", default=os.path.join(_dir, "renders"),
                    help="Directory containing per-scene render folders")
parser.add_argument("--scenes", nargs="+", default=["lego", "ship"],
                    help="Scene names to generate GIFs for")
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()

for scene in args.scenes:
    folder = os.path.join(args.render_dir, scene)

    if os.path.isdir(os.path.join(folder, scene)):
        folder = os.path.join(folder, scene)

    files = sorted(glob.glob(os.path.join(folder, 'render_*.png')))
    print(f'{scene}: found {len(files)} images in {folder}')

    if not files:
        print(f'  Skipping {scene} — no render images found')
        continue

    imgs = [imageio.v2.imread(f) for f in files]
    out = os.path.join(_dir, f'{scene}.gif')
    imageio.v2.mimsave(out, imgs, fps=args.fps)
    print(f'Saved {out}')
