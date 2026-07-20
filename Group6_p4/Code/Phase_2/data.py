from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import wxyz_to_xyzw, xyzw_to_wxyz, quat_canonical

class VIODataset(Dataset):

    def __init__(self, root_dir, frame_gap: int=1, image_size=None, mean_path=None, std_path=None, sequences=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        assert self.root_dir.is_dir(), f'{root_dir} not a directory'
        assert frame_gap >= 1, 'frame_gap must be >= 1'
        self.K = frame_gap
        self.image_size = image_size
        self.mean = None
        self.std = None
        if mean_path is not None and std_path is not None:
            self.mean = np.load(mean_path).astype(np.float32).reshape(3, 1, 1)
            self.std = np.load(std_path).astype(np.float32).reshape(3, 1, 1)
        all_seqs = sorted([p for p in self.root_dir.iterdir() if p.is_dir() and p.name.startswith('seq_')])
        if sequences is not None:
            wanted = set(sequences)
            all_seqs = [s for s in all_seqs if s.name in wanted]
        assert len(all_seqs) > 0, f'No seq_* folders found in {root_dir}'
        self.seq_paths = all_seqs
        self.index = []
        self.seq_meta = []
        for (seq_idx, seq_path) in enumerate(all_seqs):
            cfg = json.load(open(seq_path / 'config.json'))
            n_cam = cfg['n_cam_frames']
            max_i = n_cam - 1 - self.K
            assert max_i >= 0, f'frame_gap {self.K} too large for seq with {n_cam} frames'
            for i in range(max_i + 1):
                self.index.append((seq_idx, i))
            self.seq_meta.append({'path': seq_path, 'n_cam': n_cam, 'loaded': None})

    def _get_seq_arrays(self, seq_idx):
        meta = self.seq_meta[seq_idx]
        if meta['loaded'] is None:
            seq_path = meta['path']
            cam = np.load(seq_path / 'camera.npz')
            imuw = np.load(seq_path / 'imu_windows.npz')
            n_windows = len(imuw.files)
            windows = np.stack([imuw[f'arr_{k}'] for k in range(n_windows)], axis=0).astype(np.float32)
            meta['loaded'] = {'cam_positions': cam['cam_positions'].astype(np.float32), 'cam_quaternions': cam['cam_quaternions'].astype(np.float32), 'imu_windows': windows, 'images_dir': seq_path / 'images'}
        return meta['loaded']

    def _load_image(self, images_dir, frame_idx):
        img_path = images_dir / f'frame_{frame_idx:05d}.png'
        img = Image.open(img_path).convert('RGB')
        if self.image_size is not None:
            img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)

    def _compute_relative_pose(self, p_i, q_i, p_j, q_j):
        from scipy.spatial.transform import Rotation as R
        Ri = R.from_quat(wxyz_to_xyzw(q_i))
        Rj = R.from_quat(wxyz_to_xyzw(q_j))
        delta_p_world = p_j - p_i
        delta_t_body = Ri.inv().apply(delta_p_world)
        rel = Ri.inv() * Rj
        delta_q_xyzw = rel.as_quat()
        delta_q_wxyz = xyzw_to_wxyz(delta_q_xyzw)
        delta_q_wxyz = quat_canonical(delta_q_wxyz)
        return (delta_t_body.astype(np.float32), delta_q_wxyz.astype(np.float32))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        (seq_idx, i) = self.index[idx]
        j = i + self.K
        arrs = self._get_seq_arrays(seq_idx)
        img_i = self._load_image(arrs['images_dir'], i)
        img_j = self._load_image(arrs['images_dir'], j)
        img_pair = np.concatenate([img_i, img_j], axis=0).astype(np.float32)
        if self.mean is not None:
            img_pair[:3] = (img_pair[:3] - self.mean) / self.std
            img_pair[3:] = (img_pair[3:] - self.mean) / self.std
        windows = arrs['imu_windows'][i:i + self.K]
        imu = windows.reshape(-1, 6).astype(np.float32)
        p_i = arrs['cam_positions'][i]
        p_j = arrs['cam_positions'][j]
        q_i = arrs['cam_quaternions'][i]
        q_j = arrs['cam_quaternions'][j]
        (delta_t, delta_q) = self._compute_relative_pose(p_i, q_i, p_j, q_j)
        return {'img_pair': torch.from_numpy(img_pair), 'imu': torch.from_numpy(imu), 'delta_t': torch.from_numpy(delta_t), 'delta_q': torch.from_numpy(delta_q), 'seq_idx': torch.tensor(seq_idx, dtype=torch.long), 'frame_i': torch.tensor(i, dtype=torch.long)}
