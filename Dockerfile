# ─────────────────────────────────────────────────────────────────────────────
# Evidential U-Net – Deep Evidential Regression for tree height
# Base: PyTorch 2.1.2 + CUDA 12.1 + cuDNN 8 (runtime image, no build tools)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/train.py", "--config", "configs/experiments/singletask_height_unet_evidential.yaml"]
