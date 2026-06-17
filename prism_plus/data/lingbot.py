"""PRISM+ data — LingBot-Depth adapter (Robby{Sim,SimVal,Vla,Real}).

Dataset: huggingface 'robbyant/lingbot-depth' (2.71 TB, CC BY-NC-SA 4.0).
Local mount: /inspire/dataset/lingbot-mdm-depth/v1/

Depth convention:
    16-bit PNG, value in millimetres (mm). Zero = invalid / hole.
    sim/clean -> *_depth.png  (or _depth_left.png / gtdepth/)
    raw/noisy -> *_rmd2c.png  (or _rawdepth.left.png / rawdepth/)

PRISM mapping (uniform across 4 sub-datasets):
    sim_depth   <- gtdepth / *_depth.png      (clean simulator-quality)
    real_depth  <- rawdepth / *_rmd2c.png     (real sensor noisy)
    rgb         <- color / *_left.jpg         (8-bit BGR -> RGB)
    hole_mask   <- (real_depth == 0) & (sim_depth > 0)
    sensor_id   <- e.g. 'lingbot_sim_object', 'robbyvla_franka_left_405'
"""
from __future__ import annotations
import os, glob, re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shared loader
# ---------------------------------------------------------------------------

def _load_rgb(path: str, resolution: Optional[int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resolution is not None:
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def _load_depth_mm(path: str, resolution: Optional[int]) -> np.ndarray:
    """Load uint16 mm-scale depth, return float32 metres."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    if d.ndim == 3:
        d = d[..., 0]
    if resolution is not None:
        d = cv2.resize(d, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    return d.astype(np.float32) / 1000.0


def _make_sample(rgb_path: str, clean_path: str, noisy_path: str,
                  resolution: int, sensor_id: str, scene_id: str, frame_id: str) -> dict:
    rgb     = _load_rgb(rgb_path, resolution)
    clean_m = _load_depth_mm(clean_path, resolution)
    noisy_m = _load_depth_mm(noisy_path, resolution)

    # Hole = real depth missing in regions where the sim has a value
    valid_clean = clean_m > 0.0
    hole = ((noisy_m <= 0.0) & valid_clean).astype(np.float32)

    # Optional: also drop noisy values that are wildly larger (noise-token style)
    noisy_clean = noisy_m.copy()
    noisy_clean[noisy_clean <= 0.0] = 0.0

    return dict(
        rgb=torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
        sim_depth=torch.from_numpy(clean_m).unsqueeze(0),
        real_depth=torch.from_numpy(noisy_clean).unsqueeze(0),
        hole_mask=torch.from_numpy(hole).unsqueeze(0),
        sensor_id=sensor_id,
        scene_id=scene_id,
        frame_id=frame_id,
    )


# ---------------------------------------------------------------------------
# RobbySim (object_view + rrt_view) — 999K simulated paired frames
# ---------------------------------------------------------------------------

class LingBotRobbySim(Dataset):
    """Simulated paired (clean GT, noisy raw) frames from object_view/rrt_view.

    Frames are organised as <root>/{view}/<scene_id>/<cam_id>/<idx>_cam0_*.{jpg,png}.
    Each frame yields:
        * rgb       <idx>_cam0_left.jpg
        * sim_depth <idx>_cam0_depth.png (uint16 mm)
        * real_depth<idx>_cam0_rmd2c.png (uint16 mm)
    """

    def __init__(
        self,
        root: str = '/inspire/dataset/lingbot-mdm-depth/v1/RobbySim',
        views: Sequence[str] = ('object_view', 'rrt_view'),
        resolution: int = 256,
        max_samples: Optional[int] = None,
        sample_seed: int = 42,
    ):
        self.root = root
        self.resolution = int(resolution)
        self.entries: List[Tuple[str, str, str, str]] = []
        # entry = (rgb, clean, noisy, scene_token)
        for view in views:
            view_dir = os.path.join(root, view)
            if not os.path.isdir(view_dir):
                continue
            for scene in sorted(os.listdir(view_dir)):
                sd = os.path.join(view_dir, scene)
                if not os.path.isdir(sd):
                    continue
                for cam in sorted(os.listdir(sd)):
                    cd = os.path.join(sd, cam)
                    if not os.path.isdir(cd):
                        continue
                    for f in sorted(glob.glob(os.path.join(cd, '*_cam0_left.jpg'))):
                        idx = os.path.basename(f).split('_')[0]
                        clean = os.path.join(cd, f'{idx}_cam0_depth.png')
                        noisy = os.path.join(cd, f'{idx}_cam0_rmd2c.png')
                        if os.path.exists(clean) and os.path.exists(noisy):
                            self.entries.append((f, clean, noisy, f'{view}_{scene}_{cam}'))
        if not self.entries:
            raise FileNotFoundError(f'no frames under {root}/{views}')
        if max_samples is not None and max_samples < len(self.entries):
            rng = np.random.RandomState(sample_seed)
            idx = sorted(rng.choice(len(self.entries), max_samples, replace=False))
            self.entries = [self.entries[i] for i in idx]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        rgb_p, clean_p, noisy_p, scene_token = self.entries[i]
        idx = os.path.basename(rgb_p).split('_')[0]
        return _make_sample(rgb_p, clean_p, noisy_p, self.resolution,
                              sensor_id='lingbot_sim',
                              scene_id=scene_token, frame_id=idx)


# ---------------------------------------------------------------------------
# RobbySimVal — 39K validation frames (different file naming)
# ---------------------------------------------------------------------------

class LingBotRobbySimVal(Dataset):
    """<root>/val_view/<scene>/<cam>/<idx>_cam0_{rgb.left.jpg, depth_left.png, rawdepth.left.png}"""

    def __init__(
        self,
        root: str = '/inspire/dataset/lingbot-mdm-depth/v1/RobbySimVal',
        resolution: int = 256,
        max_samples: Optional[int] = None,
        sample_seed: int = 42,
    ):
        self.root = root
        self.resolution = int(resolution)
        self.entries: List[Tuple[str, str, str, str]] = []
        view_dir = os.path.join(root, 'val_view')
        for scene in sorted(os.listdir(view_dir)):
            sd = os.path.join(view_dir, scene)
            if not os.path.isdir(sd):
                continue
            for cam in sorted(os.listdir(sd)):
                cd = os.path.join(sd, cam)
                if not os.path.isdir(cd):
                    continue
                for f in sorted(glob.glob(os.path.join(cd, '*_cam0_rgb.left.jpg'))):
                    idx = os.path.basename(f).split('_')[0]
                    clean = os.path.join(cd, f'{idx}_cam0_depth_left.png')
                    noisy = os.path.join(cd, f'{idx}_cam0_rawdepth.left.png')
                    if os.path.exists(clean) and os.path.exists(noisy):
                        self.entries.append((f, clean, noisy, f'{scene}_{cam}'))
        if not self.entries:
            raise FileNotFoundError(f'no frames under {root}/val_view')
        if max_samples is not None and max_samples < len(self.entries):
            rng = np.random.RandomState(sample_seed)
            idx = sorted(rng.choice(len(self.entries), max_samples, replace=False))
            self.entries = [self.entries[i] for i in idx]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        rgb_p, clean_p, noisy_p, scene_token = self.entries[i]
        idx = os.path.basename(rgb_p).split('_')[0]
        return _make_sample(rgb_p, clean_p, noisy_p, self.resolution,
                              sensor_id='lingbot_sim_val',
                              scene_id=scene_token, frame_id=idx)


# ---------------------------------------------------------------------------
# RobbyVla — real robot manipulation, multi-sensor
# ---------------------------------------------------------------------------

class LingBotRobbyVla(Dataset):
    """<root>/<robot>/<seq>/<cam_dir>/{color,gtdepth,rawdepth}/<frame>.{jpg,png}

    cam_dir options observed: left_realsense405, right_realsense405, head_realsense435.
    Robot options: franka, ur7e.

    sensor_id format: 'robbyvla_<robot>_<cam_dir>'.
    Pass 'robots' / 'cams' to filter — useful to train one LoRA per sensor.
    """

    def __init__(
        self,
        root: str = '/inspire/dataset/lingbot-mdm-depth/v1/RobbyVla',
        robots: Sequence[str] = ('franka', 'ur7e'),
        cams: Sequence[str] = ('left_realsense405', 'right_realsense405', 'head_realsense435'),
        resolution: int = 256,
        max_samples: Optional[int] = None,
        sample_seed: int = 42,
    ):
        self.root = root
        self.resolution = int(resolution)
        self.entries: List[Tuple[str, str, str, str, str]] = []
        # entry = (rgb, clean, noisy, sensor_id, seq_id, frame_id_str)
        for robot in robots:
            rd = os.path.join(root, robot)
            if not os.path.isdir(rd):
                continue
            for seq in sorted(os.listdir(rd)):
                sd = os.path.join(rd, seq)
                if not os.path.isdir(sd):
                    continue
                for cam in cams:
                    cd = os.path.join(sd, cam)
                    if not os.path.isdir(cd):
                        continue
                    color_dir = os.path.join(cd, 'color')
                    if not os.path.isdir(color_dir):
                        continue
                    for f in sorted(glob.glob(os.path.join(color_dir, '*.jpg'))):
                        fid = os.path.splitext(os.path.basename(f))[0]
                        clean = os.path.join(cd, 'gtdepth', f'{fid}.png')
                        noisy = os.path.join(cd, 'rawdepth', f'{fid}.png')
                        if os.path.exists(clean) and os.path.exists(noisy):
                            sensor_id = f'robbyvla_{robot}_{cam}'
                            self.entries.append((f, clean, noisy, sensor_id, f'{robot}_{seq}', fid))
        if not self.entries:
            raise FileNotFoundError(f'no frames under {root}/{robots}/{cams}')
        if max_samples is not None and max_samples < len(self.entries):
            rng = np.random.RandomState(sample_seed)
            idx = sorted(rng.choice(len(self.entries), max_samples, replace=False))
            self.entries = [self.entries[i] for i in idx]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        rgb_p, clean_p, noisy_p, sensor_id, seq_id, fid = self.entries[i]
        return _make_sample(rgb_p, clean_p, noisy_p, self.resolution,
                              sensor_id=sensor_id, scene_id=seq_id, frame_id=fid)


# ---------------------------------------------------------------------------
# RobbyReal — real-world indoor, 5 sensor types (orbbec / RealSense)
# ---------------------------------------------------------------------------

_SENSOR_CANON_RE = re.compile(r'^(?P<vendor>[a-zA-Z]+)_(?P<model>[A-Za-z0-9]+)(_.*)?')


def _canon_sensor_dir(name: str) -> str:
    """Map e.g. 'orbbec_335L_CP2L8530000E' -> 'orbbec_335L'."""
    m = _SENSOR_CANON_RE.match(name)
    if not m:
        return name
    return f"{m.group('vendor')}_{m.group('model')}"


class LingBotRobbyReal(Dataset):
    """<root>/<scene>/<sensor_dir>/{color,gtdepth,rawdepth}/<frame>.{jpg,png}

    sensor_dir examples: orbbec_335_<id>, orbbec_335L_<id>, realsense_D415_<id>,
                          realsense_D435_<id>, realsense_D455_<id>.

    sensor_id format: canonical 'orbbec_335' / 'realsense_D415' (drops serial).
    Pass  to filter.
    """

    KNOWN_SENSORS = ('orbbec_335', 'orbbec_335L',
                       'realsense_D415', 'realsense_D435', 'realsense_D455')

    def __init__(
        self,
        root: str = '/inspire/dataset/lingbot-mdm-depth/v1/RobbyReal',
        sensors: Optional[Sequence[str]] = None,
        resolution: int = 256,
        max_samples: Optional[int] = None,
        sample_seed: int = 42,
    ):
        self.root = root
        self.resolution = int(resolution)
        accept = set(sensors) if sensors is not None else None
        self.entries: List[Tuple[str, str, str, str, str, str]] = []
        for scene in sorted(os.listdir(root)):
            sc_dir = os.path.join(root, scene)
            if not os.path.isdir(sc_dir):
                continue
            for sub_scene in sorted(os.listdir(sc_dir)):
                sub_dir = os.path.join(sc_dir, sub_scene)
                if not os.path.isdir(sub_dir):
                    continue
                for sd_name in sorted(os.listdir(sub_dir)):
                    sd = os.path.join(sub_dir, sd_name)
                    if not os.path.isdir(sd):
                        continue
                    canon = _canon_sensor_dir(sd_name)
                    if accept is not None and canon not in accept:
                        continue
                    color_dir = os.path.join(sd, 'color')
                    if not os.path.isdir(color_dir):
                        continue
                    for f in sorted(glob.glob(os.path.join(color_dir, '*.jpg'))):
                        fid = os.path.splitext(os.path.basename(f))[0]
                        clean = os.path.join(sd, 'gtdepth', f'{fid}.png')
                        noisy = os.path.join(sd, 'rawdepth', f'{fid}.png')
                        if os.path.exists(clean) and os.path.exists(noisy):
                            sensor_id = canon
                            self.entries.append(
                                (f, clean, noisy, sensor_id,
                                 f'{scene}_{sub_scene}_{sd_name}', fid)
                            )
        if not self.entries:
            raise FileNotFoundError(f'no frames under {root} (sensors={sensors})')
        if max_samples is not None and max_samples < len(self.entries):
            rng = np.random.RandomState(sample_seed)
            idx = sorted(rng.choice(len(self.entries), max_samples, replace=False))
            self.entries = [self.entries[i] for i in idx]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        rgb_p, clean_p, noisy_p, sensor_id, sc, fid = self.entries[i]
        return _make_sample(rgb_p, clean_p, noisy_p, self.resolution,
                              sensor_id=sensor_id, scene_id=sc, frame_id=fid)
