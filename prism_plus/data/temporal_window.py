"""PRISM+ data — Temporal sliding-window wrapper for C4 TNSM training.

Wraps any single-frame Dataset that yields (rgb, sim_depth, real_depth,
hole_mask, scene_id, frame_id) into a video-clip dataset: each sample is a
T-frame stack from the same scene_id, sorted by frame_id.

Two source modes:
    a) 'graspnet'  — GraspNet scenes (256 frames per scene, structured video)
    b) 'bytecam'   — ByteCameraDepth (assumes consecutive global IDs are
                      consecutive frames; honest video assumption only valid
                      if dataset was captured sequentially)
    c) 'dreds'     — DREDS per-scene camera rotation around an object,
                      provides 30-frame multi-view sequences (NOT true video
                      but spatially-coherent multi-view, good for ConvGRU
                      regularisation when no real video is available)
"""
from __future__ import annotations
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class TemporalWindow(Dataset):
    """Returns a T-frame stack from the same scene.

    Each item yields a dict whose tensor fields are stacked along dim=0:
        rgb         [T, 3, H, W]
        sim_depth   [T, 1, H, W]
        real_depth  [T, 1, H, W]
        hole_mask   [T, 1, H, W]
        scene_id    str    (single — shared by all T frames)
        frame_ids   list[str]
        sensor_id   str    (single)
    """

    def __init__(
        self,
        base: Dataset,
        T: int = 4,
        stride: int = 1,
        min_frames_per_scene: int = 4,
    ):
        if T < 2:
            raise ValueError('T must be >= 2 for temporal windows')
        self.base = base
        self.T = int(T)
        self.stride = int(stride)

        # Group base indices by scene_id, sorted by frame_id within scene.
        # Try to use base.entries to avoid loading the whole sample;
        # else fall back to per-index __getitem__ (slow but correct).
        scene_groups: dict = defaultdict(list)
        entries = getattr(base, 'entries', None)
        if entries is not None and isinstance(entries[0], tuple):
            # DREDS-style: entries = [(scene_dir, frame_id), ...]
            from pathlib import Path as _P
            for idx, (scene_dir, fid) in enumerate(entries):
                sid = _P(scene_dir).name
                scene_groups[sid].append((fid, idx))
        else:
            for idx in range(len(base)):
                sample = base[idx]
                sid = sample.get('scene_id', 'default')
                fid = sample.get('frame_id', str(idx))
                scene_groups[sid].append((fid, idx))

        # For each scene with >= min_frames, enumerate valid window starts
        self.windows: List[Sequence[int]] = []
        self.scene_ids: List[str] = []
        for sid, items in scene_groups.items():
            items = sorted(items, key=lambda x: x[0])
            indices = [i for _, i in items]
            if len(indices) < min_frames_per_scene:
                continue
            for start in range(0, len(indices) - (T - 1) * stride, 1):
                window = [indices[start + j * stride] for j in range(T)]
                self.windows.append(window)
                self.scene_ids.append(sid)

        if not self.windows:
            raise RuntimeError(f'No valid {T}-frame windows found in base dataset')

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        indices = self.windows[idx]
        samples = [self.base[i] for i in indices]

        out: dict = {}
        # Stack tensor fields
        for key in ('rgb', 'sim_depth', 'real_depth', 'hole_mask'):
            if key in samples[0]:
                out[key] = torch.stack([s[key] for s in samples], dim=0)
        out['scene_id']  = self.scene_ids[idx]
        out['frame_ids'] = [s.get('frame_id', str(i)) for s, i in zip(samples, indices)]
        out['sensor_id'] = samples[0].get('sensor_id', 'unknown')
        return out


class ByteCamConsecutiveAdapter(Dataset):
    """Wraps a ByteCameraDepth-style flat dataset and assigns a synthetic
    scene_id by floor(frame_id_int / window_per_scene) so that TemporalWindow
    can find consecutive groups.

    USE WITH CAUTION: only meaningful if the underlying capture was sequential.
    """

    def __init__(self, base: Dataset, window_per_scene: int = 50):
        self.base = base
        self.window_per_scene = int(window_per_scene)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = dict(self.base[idx])
        # Use idx-derived synthetic scene_id
        sample['scene_id'] = f'bytecam_seq_{idx // self.window_per_scene:04d}'
        sample['frame_id'] = f'{idx:08d}'
        sample.setdefault('sensor_id', 'bytecam_realsense')
        return sample
