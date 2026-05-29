"""Vision Foundation Model (VFM) backbone for PRISM+.

Wraps MoGe2 (DINOv2-Large) and DINOv2 as semantic feature extractors.
"""
from .interface import (
    VFMInterface,
    MoGe2VFM,
    DINOv2VFM,
    VFMProjector,
    create_vfm,
)

__all__ = ["VFMInterface", "MoGe2VFM", "DINOv2VFM", "VFMProjector", "create_vfm"]
