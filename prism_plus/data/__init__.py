"""Datasets for PRISM+."""
from .bytecam_depth import ByteCamDepthDataset
from .gt_decompose import decompose_gt, decompose_gt_preset, PRESETS, DEFAULT_PRESET
from .gt_decompose_wrapper import GTDecomposeWrapper

__all__ = [
    "ByteCamDepthDataset",
    "decompose_gt", "decompose_gt_preset", "PRESETS", "DEFAULT_PRESET",
    "GTDecomposeWrapper",
]

# C3 / C4 new adapters
try:
    from .dreds            import DREDSDataset
    from .multi_sensor     import MultiSensorDataset, SENSOR_REGISTRY, make_loader
    from .temporal_window  import TemporalWindow, ByteCamConsecutiveAdapter
except ImportError as _e:
    DREDSDataset = MultiSensorDataset = TemporalWindow = ByteCamConsecutiveAdapter = None
