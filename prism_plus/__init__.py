"""PRISM+ — Physics-Prior Closure for Consistent and Generalizable Depth Noise Modeling.

TPAMI extension of PRISM (ICML 2026), with four targeted improvements:
    * C1: Spatial-SPR BND (multi-scale VFM cross-attention)
    * C2: M-conditioned NRG (mask-conditioned ControlNet) [TODO]
    * C3: LoRA-SPA (rank-4 LoRA for sensor-agnostic generalization) [TODO]
    * C4: TNSM (flow-guided ConvGRU for temporal coherence) [TODO]
"""
from . import models, data, losses, metrics
__version__ = "0.1.0"
__all__ = ["models", "data", "losses", "metrics"]
