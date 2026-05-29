# PRISM+: Physics-Prior Closure for Consistent and Generalizable Depth Noise Modeling

> **TPAMI extension of PRISM (ICML 2026)** — closing four identifiable gaps of the
> original PRISM framework along intra-model and generalization axes.

## What is PRISM+?

PRISM (ICML 2026) introduced bimodal depth noise disentanglement (sensing-invalidation +
measurement-inaccuracy) via a tripartite SPR / BND / NRG architecture. PRISM+ extends it
with four targeted improvements:

| ID | Improvement | Status |
|----|-------------|--------|
| **C1** | **Spatial-SPR BND** — multi-scale VFM cross-attention on encoder skip features (replaces GAP-based channel attention) | ✅ Implemented |
| **C2** | **M-conditioned NRG** — mask-conditioned ControlNet with boundary loss (eliminates flying-pixel artifacts) | 🚧 Stage 2 |
| **C3** | **LoRA-SPA** — rank-4 LoRA for sensor-agnostic generalization (ToF / LiDAR transfer) | 🚧 Stage 3 |
| **C4** | **TNSM** — flow-guided ConvGRU for temporal coherence on video sequences | 🚧 Stage 4 |

All improvements add **< 3M new params** (< 0.7 % of the PRISM base).

## Repository layout

```
prism_plus/
├── prism_plus/                  # Python package
│   ├── models/
│   │   ├── bnd.py               # PRISM baseline BND
│   │   ├── bnd_spatial.py       # PRISM+ C1: SpatialBND
│   │   ├── nrg.py               # PRISM baseline NRG (Stage B)
│   │   ├── mnrg.py              # PRISM+ C2 (TODO)
│   │   ├── lora_spa.py          # PRISM+ C3 (TODO)
│   │   ├── tnsm.py              # PRISM+ C4 (TODO)
│   │   └── vfm/                 # MoGe2 / DINOv2 backbone
│   ├── data/
│   │   └── bytecam_depth.py     # ByteCameraDepth dataset
│   ├── losses/
│   │   └── hpps.py              # H-PPS loss
│   ├── diffusion/               # ControlNet + Latent Diffusion (Stage B)
│   ├── metrics.py               # PRISM + PRISM+ evaluation metrics
│   └── utils/
├── tools/                       # CLI entrypoints
│   └── train_bnd.py
├── configs/                     # YAML configs (one per stage)
│   └── stage1_bnd_spatial.yaml
├── scripts/                     # Shell launch scripts
│   └── train_stage1.sh
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
# Editable install (recommended for development)
git clone https://github.com/ShawnLeung-SII/PRISM-Plus.git
cd PRISM-Plus
pip install -e .

# Optional: diffusion (Stage 2+ NRG)
pip install -e ".[diffusion]"
```

## Training

### Stage 1 — SpatialBND (C1)

```bash
# Single GPU debug
python tools/train_bnd.py --config configs/stage1_bnd_spatial.yaml --debug

# 4×H200 DDP (provided launch script)
bash scripts/train_stage1.sh
```

### Inspire (Qizhi) cluster submission

For users on the Inspire cluster, submit via `qz train CreateJob`:

```bash
qz train CreateJob --data '{
    "workspace_id":           "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6",
    "project_id":             "project-6ee538a7-4c8b-410c-9653-77111474b204",
    "logic_compute_group_id": "lcg-df089db8-817a-4aa8-a164-eb1a32948564",
    "name":                   "prism_plus_stage1",
    "framework":              "pytorch",
    "task_priority":          5,
    "command":                "bash /inspire/ssd/.../prism_plus/scripts/train_stage1.sh",
    "framework_config": [{
        "image_type":     "custom",
        "image":          "docker.sii.shaipower.online/base/ngc-pytorch:25.02-cuda12.8.0-py3",
        "instance_count": 1,
        "spec_id":        "4dd0e854-e2a4-4253-95e6-64c13f0b5117",
        "shm_gi":         120
    }]
}'
```

## Data

ByteCameraDepth — RealSense D435 paired (RGB / sim_depth / real_depth / hole_mask).

Expected layout under `data_root`:
```
rgb/, sim_depth/, real_depth/, processed/hole_mask/, splits/{train,val,test}.txt
```

## Citation

If you use PRISM or PRISM+, please cite:

```bibtex
@inproceedings{liang2026prism,
  title  = {{PRISM}: Learning Realistic Depth via Physics-Grounded Noise
            Disentanglement with Semantic-Geometric Collaboration},
  author = {Liang, Xiujian and Liu, Jiacheng and Sun, Mingyang and He, Qichen
            and Cheng, Anda and Lu, Cewu and Sun, Jianhua},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year   = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
