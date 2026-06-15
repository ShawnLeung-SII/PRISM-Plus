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

# C3 LoRA-SPA + C4 TNSM + flow utils
try:
    from .lora_spa  import LoRASPA, LoRAAdapter, create_lora_spa
    _HAS_LORA = True
except ImportError:
    LoRASPA = LoRAAdapter = create_lora_spa = None
    _HAS_LORA = False

try:
    from .tnsm import TNSM, ConvGRUCell
    _HAS_TNSM = True
except ImportError:
    TNSM = ConvGRUCell = None
    _HAS_TNSM = False

try:
    from .flow_utils import RAFTFlow, RAFTWrapper, backward_warp, downsample_flow, make_zero_flow
    _HAS_FLOW = True
except ImportError:
    RAFTFlow = RAFTWrapper = backward_warp = downsample_flow = make_zero_flow = None
    _HAS_FLOW = False
