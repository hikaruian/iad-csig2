#!/usr/bin/env bash
# Train one INP-Former per group (default 5 classes) via existing train.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GROUP_DATA="${GROUP_DATA:-$ROOT/data/groups}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/runs/groups}"
EPOCHS="${EPOCHS:-150}"
GPUS="${GPUS:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ ! -d "$GROUP_DATA" ]]; then
  echo "Run scripts/make_group_links.sh first" >&2
  exit 1
fi


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

#mapfile -t GROUPS < <(find "$GROUP_DATA" -mindepth 1 -maxdepth 1 -type d -name 'g*' -printf '%f\n' | sort)
#GROUPS=()
#while IFS= read -r line; do
#    GROUPS+=("$line")
#done < <(


find "$GROUP_DATA" -mindepth 1 -maxdepth 1 -type d -name 'g*' -printf '%f\n' | sort > /tmp/groups.tmp


GROUPS1=()
while IFS= read -r line; do
    echo "line:$line"
    GROUPS1+=("$line")
done < /tmp/groups.tmp
rm -f /tmp/groups.tmp

echo "training ${#GROUPS1[@]} groups  epochs=$EPOCHS  gpus=$GPUS"


k=0
for gid in "${GROUPS1[@]}"; do
  k=$((k + 1))
  train_root="$GROUP_DATA/$gid/Train"
  save_dir="$SAVE_ROOT/$gid"
  ckpt="$save_dir/model.pth"
  if [[ "$SKIP_EXISTING" == "1" && -f "$ckpt" ]]; then
    echo "[$k/${#GROUPS1[@]}] skip $gid (found $ckpt)"
    continue
  fi
  echo "[$k/${#GROUPS1[@]}] TRAIN $gid  classes=$(ls "$train_root" | tr '\n' ' ')"
  mkdir -p "$save_dir"
  if [[ "$GPUS" -gt 1 ]]; then
    torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" train.py \
      --train-root "$train_root" \
      --save-dir "$save_dir" \
      --encoder dinov2reg_vit_large_14 \
      --image-size 448 --inp-num 6 \
      --epochs "$EPOCHS" \
      --batch-size 4 --grad-accum 2 \
      --num-workers 2 --amp
  else
    python train.py \
      --train-root "$train_root" \
      --save-dir "$save_dir" \
      --encoder dinov2reg_vit_large_14 \
      --image-size 448 --inp-num 6 \
      --epochs "$EPOCHS" \
      --batch-size 4 --grad-accum 2 \
      --num-workers 2 --amp
  fi
done

echo "done. $SAVE_ROOT/<gid>/model.pth"

