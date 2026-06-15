"""PRISM+ data — Unified multi-sensor dataset router.

For C3 LoRA-SPA training we mix samples from multiple sensors in one loader
(or maintain per-sensor loaders) and route each sample's gradient to its
corresponding LoRA adapter via the 'sensor_id' field.

Source sensors currently supported:
    'bytecam_realsense'   — ByteCameraDepth (project dataset)
    'dreds_d415'          — DREDS, RealSense D415 stereo noise model
    'dreds_l515'          — DREDS, RealSense L515 ToF noise model (TODO: novel)
    'graspnet_kinect'     — GraspNet Azure Kinect (ToF)            (TODO: when downloaded)
    'graspnet_realsense'  — GraspNet RealSense                     (TODO: when downloaded)
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset


SENSOR_REGISTRY: Dict[str, str] = {
    'bytecam_realsense': 'ByteCameraDepth (RealSense, project source domain)',
    'dreds_d415':        'DREDS RealSense D415 simulated noise',
    'dreds_l515':        'DREDS RealSense L515 ToF simulated noise',
    'graspnet_kinect':   'GraspNet Azure Kinect ToF',
    'graspnet_realsense':'GraspNet RealSense',
}


class MultiSensorDataset(Dataset):
    """Concatenates several per-sensor datasets, preserving sensor_id in each sample.

    Each child dataset must yield a dict containing a 'sensor_id' field. We
    do not modify the samples — only forward  / .
    The optional  enables weighted sampling per sensor at
    the *DataLoader* level (build a WeightedRandomSampler from these).
    """

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        sensor_id_weights: Optional[Dict[str, float]] = None,
    ):
        if not datasets:
            raise ValueError('need at least one sensor dataset')
        self.datasets = datasets
        self.order = list(datasets.keys())
        self._concat = ConcatDataset([datasets[k] for k in self.order])
        # Build sample → sensor_id map for sampler weighting
        self._idx_to_sensor = []
        for sid in self.order:
            self._idx_to_sensor.extend([sid] * len(datasets[sid]))
        self.sensor_id_weights = sensor_id_weights or {sid: 1.0 for sid in self.order}

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int):
        return self._concat[idx]

    def sample_weights(self) -> torch.Tensor:
        return torch.tensor(
            [self.sensor_id_weights.get(s, 1.0) for s in self._idx_to_sensor],
            dtype=torch.float32,
        )


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
    use_sensor_weights: bool = False,
):
    from torch.utils.data import DataLoader, WeightedRandomSampler
    sampler = None
    if shuffle and use_sensor_weights and hasattr(dataset, 'sample_weights'):
        weights = dataset.sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)
