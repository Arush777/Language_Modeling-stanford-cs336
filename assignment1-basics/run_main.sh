#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config (edit these)
# ----------------------------
TRAIN_BIN="data/tinystories_train.uint32.bin"
VALID_BIN="data/tinystories_valid.uint32.bin"
VOCAB_SIZE=10000

RUN_NAME="tinystories_5k"

CONTEXT_LENGTH=256
BATCH_SIZE=16

D_MODEL=256
NUM_LAYERS=4
NUM_HEADS=8
D_FF=1024
ROPE_THETA=10000.0

MAX_ITERS=5000
LOG_EVERY=25
EVAL_EVERY=500
EVAL_ITERS=50

LR=3e-4
MIN_LR=3e-5
WARMUP_ITERS=500
COSINE_CYCLE_ITERS=5000
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0




# Apple Silicon: usually fastest is "mps"
DEVICE="cpu"

# ----------------------------
# Derived paths
# ----------------------------
OUT_DIR="runs/${RUN_NAME}"
CKPT_PATH="${OUT_DIR}/checkpoint.pt"
LOG_PATH="${OUT_DIR}/checkpoint.trainlog.jsonl"

mkdir -p "${OUT_DIR}"

echo "Run: ${RUN_NAME}"
echo "Out dir: ${OUT_DIR}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Log: ${LOG_PATH}"
echo "Device: ${DEVICE}"

# ----------------------------
# Train (conditionally pass --log_path if your main.py supports it)
# ----------------------------



EXTRA_ARGS=()
if uv run python main.py -h 2>/dev/null | grep -q -- "--log_path"; then
  EXTRA_ARGS+=( --log_path "${LOG_PATH}" )
fi

RESUME=1
if [[ "${RESUME}" -eq 1 ]]; then
  EXTRA_ARGS+=( --resume )
fi



uv run python main.py \
  --train_bin "${TRAIN_BIN}" \
  --valid_bin "${VALID_BIN}" \
  --vocab_size "${VOCAB_SIZE}" \
  --context_length "${CONTEXT_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --d_model "${D_MODEL}" \
  --num_layers "${NUM_LAYERS}" \
  --num_heads "${NUM_HEADS}" \
  --d_ff "${D_FF}" \
  --rope_theta "${ROPE_THETA}" \
  --lr "${LR}" \
  --min_lr "${MIN_LR}" \
  --warmup_iters "${WARMUP_ITERS}" \
  --cosine_cycle_iters "${COSINE_CYCLE_ITERS}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --grad_clip "${GRAD_CLIP}" \
  --max_iters "${MAX_ITERS}" \
  --log_every "${LOG_EVERY}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_iters "${EVAL_ITERS}" \
  --ckpt_path "${CKPT_PATH}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"



# ----------------------------
# Plot loss curve (requires JSONL log)
# ----------------------------
if [[ ! -f "${LOG_PATH}" ]]; then
  echo "No train log found. Add JSONL logging to main.py (train/val loss records) to auto-plot."
  echo "Expected one of: runs/.../trainlog.jsonl or checkpoint.trainlog.jsonl"
  exit 0
fi

echo "Plotting from log: ${LOG_PATH}"

uv run python - <<PY
import json, math
from pathlib import Path

log_path = Path("${LOG_PATH}")

if not log_path.exists():
    raise FileNotFoundError(f"Log file not found: {log_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

train_it, train_loss = [], []
val_it, val_loss = [], []

with log_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        it = r.get("iter")
        loss = r.get("loss")
        split = r.get("split")
        if it is None or loss is None or split is None:
            continue
        if split == "train":
            train_it.append(it); train_loss.append(loss)
        elif split in ("val", "valid", "validation"):
            val_it.append(it); val_loss.append(loss)

train = sorted(zip(train_it, train_loss))
val = sorted(zip(val_it, val_loss))

out_dir = log_path.parent

# Loss curve
plt.figure()
if train:
    x,y = zip(*train); plt.plot(x,y,label="train loss")
if val:
    x,y = zip(*val); plt.plot(x,y,label="val loss")
plt.xlabel("iteration"); plt.ylabel("loss"); plt.title("Loss curves"); plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "loss_curve.png", dpi=200)

# PPL curve
plt.figure()
if train:
    x,y = zip(*train); plt.plot(x,[math.exp(v) for v in y],label="train ppl")
if val:
    x,y = zip(*val); plt.plot(x,[math.exp(v) for v in y],label="val ppl")
plt.xlabel("iteration"); plt.ylabel("perplexity"); plt.title("Perplexity curves"); plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "ppl_curve.png", dpi=200)

print("Saved:", out_dir / "loss_curve.png")
print("Saved:", out_dir / "ppl_curve.png")
PY
