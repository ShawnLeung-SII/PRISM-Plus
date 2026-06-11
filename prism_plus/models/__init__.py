"""PRISM+ models — BND, NRG and PRISM+ extensions."""
from .bnd import BND, create_bnd
from .bnd_spatial import SpatialBND, create_spatial_bnd       # v0.1.0 (baseline C1)
from .bnd_plus    import PRISMPlusBND, create_bnd_plus, GatedVFMCrossAttn  # v0.2.0
from .boundary    import BoundaryBranch, BoundaryRefiner

# NRG is optional (requires diffusion stack: cldm + ldm + diffusers)
try:
    from .nrg            import NRG
    from .nrg_standalone import NRGStandalone   # PRISM+ C2 (independent NRG)
    _HAS_NRG = True
except ImportError:
    NRG = None
    NRGStandalone = None
    _HAS_NRG = False

__all__ = [
    # Baseline PRISM
    "BND", "create_bnd",
    # v0.1.0 C1-naive (kept for ablation)
    "SpatialBND", "create_spatial_bnd",
    # v0.2.0 C1-redesigned (Coarse Semantic Prior + Boundary Refinement)
    "PRISMPlusBND", "create_bnd_plus", "GatedVFMCrossAttn",
    "BoundaryBranch", "BoundaryRefiner",
    # NRG (Stage 2 / PRISM+ C2)
    "NRG", "NRGStandalone",
]
