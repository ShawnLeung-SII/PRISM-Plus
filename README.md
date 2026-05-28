# PRISM+: Physics-Prior Closure for Consistent and Generalizable Depth Noise Modeling

**TPAMI Extension of PRISM (ICML 2026)**

PRISM+ extends the PRISM framework with four targeted improvements:
- **C1 Spatial-SPR**: Multi-scale VFM cross-attention for pixel-level material-noise alignment
- **C2 M-conditioned NRG**: Mask-conditioned ControlNet for flying-pixel elimination  
- **C3 LoRA-SPA**: Rank-4 LoRA for sensor-agnostic generalization
- **C4 TNSM**: Flow-guided ConvGRU for temporal coherence

## Project Structure
```
prism_plus/
├── models/
│   ├── spatial_spr_bnd.py     # C1: Spatial-SPR BND (extends PRISM V9)
│   ├── mnrg.py                # C2: M-conditioned NRG (TODO)
│   ├── lora_spa.py            # C3: LoRA-SPA (TODO)
│   └── tnsm.py                # C4: Temporal noise state module (TODO)
├── utils/
│   └── metrics_prism_plus.py  # New metrics: boundary_mae, flying_pixel_rate, TNFR
├── configs/
│   └── stage1_spatial_spr.yaml
├── scripts/
│   └── train_stage1.sh        # H200 training script
└── train_stage1_bnd.py        # Stage 1 training entrypoint
```

## Dependencies
- Base: PRISM codebase at `latpixdepth/`
- Python 3.10, PyTorch 2.9.1+cu128
- Conda env: `latpixdepth` (on HDD persistent storage)

## Training
Stage 1 (Spatial-SPR BND, 4×H200):
```bash
qz train CreateJob --data @configs/qz_stage1.json  # via Inspire-CLI
```
