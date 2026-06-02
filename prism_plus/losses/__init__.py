"""Loss functions for PRISM+ (BND H-PPS, NRG boundary, ...)."""
from .hpps import HPPSLoss, positive_weighted_bce, dice_loss, pp_dhm_bce
from .boundary import (
    BNDPlusLoss,
    Sobel, sobel_magnitude,
    mask_boundary, boundary_band,
    edge_bce_dice, band_weighted_bce, signed_distance_loss,
)

__all__ = [
    # H-PPS (baseline)
    "HPPSLoss", "positive_weighted_bce", "dice_loss", "pp_dhm_bce",
    # Boundary-aware (new for v0.2.0)
    "BNDPlusLoss",
    "Sobel", "sobel_magnitude",
    "mask_boundary", "boundary_band",
    "edge_bce_dice", "band_weighted_bce", "signed_distance_loss",
]
