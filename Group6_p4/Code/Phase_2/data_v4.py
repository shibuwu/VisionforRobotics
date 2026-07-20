from pathlib import Path
import numpy as np
import torch
from data import VIODataset

class VIODatasetV4(VIODataset):

    def _get_seq_arrays(self, seq_idx):
        meta = self.seq_meta[seq_idx]
        if meta['loaded'] is None or 'h4pt_gt' not in meta['loaded']:
            super()._get_seq_arrays(seq_idx)
            seq_path = meta['path']
            homo_path = seq_path / 'homography_gt.npz'
            if not homo_path.exists():
                raise FileNotFoundError(f'Missing {homo_path}. Run compute_homography_gt.py first.')
            homo = np.load(homo_path)
            meta['loaded']['h4pt_gt'] = homo['h4pt'].astype(np.float32)
            meta['loaded']['H_gt'] = homo['H'].astype(np.float32)
        return meta['loaded']

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        (seq_idx, i) = self.index[idx]
        arrs = self._get_seq_arrays(seq_idx)
        if self.K == 1:
            h4pt = arrs['h4pt_gt'][i]
            H = arrs['H_gt'][i]
        else:
            H = np.eye(3, dtype=np.float32)
            for k in range(self.K):
                H = arrs['H_gt'][i + k] @ H
            if abs(H[2, 2]) > 1e-12:
                H = H / H[2, 2]
            from compute_homography_gt import homography_to_4point
            (h_img, w_img) = (240, 320) if self.image_size is None else self.image_size
            h4pt = homography_to_4point(H, (h_img, w_img))
        item['h4pt_gt'] = torch.from_numpy(h4pt.astype(np.float32))
        item['H_gt'] = torch.from_numpy(H.astype(np.float32))
        return item
