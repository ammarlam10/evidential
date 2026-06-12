# Evidential U-Net for Tree Height

Pixel-wise **mean tree height** regression with uncertainty estimates using [Deep Evidential Regression](https://www.mit.edu/~amini/pubs/pdf/deep-evidential-regression.pdf) (Amini et al., NeurIPS 2020).

The model is a U-Net with ResNet50 encoder. The final head outputs Normal Inverse-Gamma (NIG) hyperparameters `(γ, ν, α, β)` per pixel. At inference, `γ` is the point prediction and derived maps give aleatoric and epistemic uncertainty.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU (16 GB+ VRAM recommended)
- Zarr dataset mounted at `/data/4g.zarr` (see config paths)

## Setup

### 1. Configure paths

Edit `docker-compose.yml` volume mounts if your dataset or repo live elsewhere.

### 2. Build the image

```bash
docker compose build
```

### 3. Compute normalisation statistics (run once)

```bash
docker compose run --rm compute_stats
# Output: artifacts/norm_stats.json
```

## Training

**Single-task height (recommended):**

```bash
docker compose run --rm train
```

**Multi-task (tree count + height):**

```bash
docker compose run --rm train_multitask
```

**8-GPU seed ensemble (host script):**

```bash
bash scripts/experiments/run_evidential_height_8gpu.sh
```

Checkpoints are written under `artifacts/experiments/singletask_height_unet_evidential/` (or paths set via `--run-dir`).

## Evaluation

```bash
docker compose run --rm eval
```

## Project structure

```
evidential/
├── configs/
│   ├── unet_evidential.yaml                      # multi-task (count + height)
│   └── experiments/singletask_height_unet_evidential.yaml
├── src/
│   ├── data/           # Zarr-backed BiomassDataset
│   ├── losses/         # EvidentialRegressionLoss (NIG)
│   ├── models/         # UNetEvidential
│   └── training/       # Trainer with evidential metrics
└── scripts/
    ├── train.py
    ├── evaluate.py
    ├── compute_stats.py
    └── experiments/run_evidential_height_8gpu.sh
```
