"""PRISM+ models — BND, NRG and PRISM+ extensions."""
from .bnd import BND, create_bnd
from .bnd_spatial import SpatialBND, create_spatial_bnd

# NRG is optional (requires diffusion stack: cldm + ldm + diffusers)
try:
    from .nrg import NRG
    _HAS_NRG = True
except ImportError as _e:
    NRG = None
    _HAS_NRG = False

__all__ = [
    "BND", "create_bnd",
    "SpatialBND", "create_spatial_bnd",
    "NRG",
]
