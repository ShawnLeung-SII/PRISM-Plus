"""PRISM+ data — DREDS (ECCV 2022) adapter for PRISM batch protocol.

DREDS naming                     PRISM naming
────────────────────────   ──────────────────
_depth_120.exr  (clean GT)    sim_depth    (noise-free simulator output)
_simDepthImage.exr (noisy)    real_depth   (RealSense noise model output)
derived from noisy invalid    hole_mask    (where the sim falls apart)

Scene layout:
    DREDS-CatKnown/shapenet_generate_1216/{val_part2,train_part*,test}/
    └── <scene_id>/                      # 100 scenes/val_part2
        └── <fid>_color.png             # 360x640 RGB
            <fid>_depth_120.exr         # clean depth (metres)
            <fid>_simDepthImage.exr     # noisy depth (metres)
            <fid>_mask.exr / _meta.txt
"""
from __future__ import annotations
import glob, os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import OpenEXR, Imath
    _HAS_EXR = True
except Exception:
    _HAS_EXR = False


def _read_exr_r(path: str) -> np.ndarray:
    f = OpenEXR.InputFile(path)
    dw = f.header()['dataWindow']
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    return np.frombuffer(f.channel('R', pt), dtype=np.float32).reshape(h, w).copy()


def _list_all_frames(root: str, splits: Sequence[str]) -> list:
    out = []
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for scene in sorted(os.listdir(split_dir)):
            sd = os.path.join(split_dir, scene)
            if not os.path.isdir(sd):
                continue
            for f in sorted(glob.glob(os.path.join(sd, '*_color.png'))):
                fid = os.path.basename(f).split('_')[0]
                out.append((sd, fid))
    return out


class DREDSDataset(Dataset):
    """DREDS dataset adapter conforming to the PRISM batch protocol.

    Sample dict:
        rgb         [3, H, W]  float32 in [0,1]
        sim_depth   [1, H, W]  metres, 0 where invalid
        real_depth  [1, H, W]  metres, 0 where invalid
        hole_mask   [1, H, W]  {0,1} where real depth is missing/spurious
        sensor_id   str
        scene_id, frame_id  str
    """

    def __init__(
        self,
        root: str,
        splits: Sequence[str] = ('shapenet_generate_1216/val_part2',),
        resolution: Optional[int] = 256,
        sensor_id: str = 'dreds_d415',
        max_samples: Optional[int] = None,
        sample_seed: int = 42,
        depth_unit_scale: float = 1.0,
        invalid_thresh_ratio: float = 1.5,
    ):
        if not _HAS_EXR:
            raise ImportError('OpenEXR + Imath required; pip install OpenEXR Imath')
        self.root = root
        self.splits = list(splits)
        self.resolution = int(resolution) if resolution else None
        self.sensor_id = sensor_id
        self.depth_unit_scale = float(depth_unit_scale)
        self.invalid_thresh_ratio = float(invalid_thresh_ratio)

        all_frames = _list_all_frames(root, splits)
        if not all_frames:
            raise FileNotFoundError(f'no frames found under {root}/{splits}')
        if max_samples is not None and max_samples < len(all_frames):
            rng = np.random.RandomState(sample_seed)
            idx = rng.choice(len(all_frames), max_samples, replace=False)
            all_frames = [all_frames[i] for i in sorted(idx)]
        self.entries = all_frames

    def __len__(self) -> int:
        return len(self.entries)

    def _resize(self, x: np.ndarray, mode: str) -> np.ndarray:
        if self.resolution is None:
            return x
        interp = {'rgb': cv2.INTER_AREA, 'depth': cv2.INTER_NEAREST,
                  'mask': cv2.INTER_NEAREST}[mode]
        return cv2.resize(x, (self.resolution, self.resolution), interpolation=interp)

    def __getitem__(self, idx: int) -> dict:
        scene_dir, fid = self.entries[idx]

        rgb = cv2.imread(os.path.join(scene_dir, f'{fid}_color.png'), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = self._resize(rgb, 'rgb').astype(np.float32) / 255.0

        clean = _read_exr_r(os.path.join(scene_dir, f'{fid}_depth_120.exr')) * self.depth_unit_scale
        noisy = _read_exr_r(os.path.join(scene_dir, f'{fid}_simDepthImage.exr')) * self.depth_unit_scale
        clean = self._resize(clean, 'depth')
        noisy = self._resize(noisy, 'depth')

        valid_clean = clean > 0
        too_large = (noisy > clean * self.invalid_thresh_ratio) & valid_clean
        hole = ((noisy <= 0) | too_large) & valid_clean
        hole = hole.astype(np.float32)

        noisy_clean = noisy.copy()
        noisy_clean[hole > 0.5] = 0.0
        noisy_clean[noisy_clean <= 0] = 0.0

        return dict(
            rgb=torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
            sim_depth=torch.from_numpy(clean).unsqueeze(0),
            real_depth=torch.from_numpy(noisy_clean).unsqueeze(0),
            hole_mask=torch.from_numpy(hole).unsqueeze(0),
            sensor_id=self.sensor_id,
            scene_id=Path(scene_dir).name,
            frame_id=fid,
        )
