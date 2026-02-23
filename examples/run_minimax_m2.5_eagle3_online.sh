#!/bin/bash

# ============================================================
# 1) Resolve Paths
# ============================================================
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ============================================================
# 2) Runtime Environment
# ============================================================
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/cache/compiled_kernels"
export PYTHONPATH="$ROOT_DIR/scripts:${PYTHONPATH:-}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"

# ============================================================
# 3) Launch Parameters
# ============================================================
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}
TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/home/scratch.fengyul_coreai/model_ckpt/MiniMax-M2.5}
TARGET_MODEL_BACKEND=${TARGET_MODEL_BACKEND:-sglang}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-triton}
SGLANG_EP_SIZE=${SGLANG_EP_SIZE:-$TP_SIZE}

# ============================================================
# 4) Training Launch
# ============================================================
torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --trust-remote-code \
    --target-model-backend "$TARGET_MODEL_BACKEND" \
    --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND" \
    --sglang-ep-size "$SGLANG_EP_SIZE" \
    --draft-model-config "$ROOT_DIR/configs/minimax-m2.5-eagle3.json" \
    --train-data-path "$ROOT_DIR/cache/dataset/ultrachat_train.jsonl" \
    --build-dataset-num-proc "$BUILD_DATASET_NUM_PROC" \
    --output-dir "$ROOT_DIR/outputs/minimax-m2.5-eagle3-ultrachat" \
    --num-epochs 2 \
    --batch-size 1 \
    --tp-size "$TP_SIZE" \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template minimax-m2 \
    --cache-dir "$ROOT_DIR/cache" \
    --report-to tensorboard \
    --log-interval 10
