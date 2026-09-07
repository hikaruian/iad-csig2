#!/usr/bin/env bash
# One infer.py call per group (all classes in that group share one ckpt).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GROUP_DATA="${GROUP_DATA:-$ROOT/data/groups}"
#SAVE_ROOT="${SAVE_ROOT:-/home/runs/groups}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/runs/groups}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/groups}"
SIGMA="${SIGMA:-9.5}"

if [[ ! -d "$GROUP_DATA" ]]; then
  echo "Run scripts/make_group_links.sh first" >&2
  exit 1
fi

#mapfile -t GROUPS < <(find "$GROUP_DATA" -mindepth 1 -maxdepth 1 -type d -name 'g*' -printf '%f\n' | sort)


find "$GROUP_DATA" -mindepth 1 -maxdepth 1 -type d -name 'g*' -printf '%f\n' | sort > /tmp/groups1.tmp


GROUPS1=()
while IFS= read -r line; do
    echo "line:$line"
    GROUPS1+=("$line")
done < /tmp/groups1.tmp
rm -f /tmp/groups1.tmp


echo "infer ${#GROUPS1[@]} groups  sigma=$SIGMA"

k=0
ok=0
for gid in "${GROUPS1[@]}"; do
  k=$((k + 1))
  ckpt=""
  for cand in "$SAVE_ROOT/$gid/best.pth" "$SAVE_ROOT/$gid/model.pth" "$SAVE_ROOT/$gid/last.pth"; do
    if [[ -f "$cand" ]]; then
      ckpt="$cand"
      break
    fi
  done
  test_root="$GROUP_DATA/$gid/Test_A"
  if [[ -z "$ckpt" ]]; then
    echo "[$k/${#GROUPS1[@]}] SKIP $gid (no ckpt)"
    continue
  fi
  if [[ ! -d "$test_root" ]]; then
    echo "[$k/${#GROUPS1[@]}] SKIP $gid (no Test_A)"
    continue
  fi
  echo "[$k/${#GROUPS1[@]}] INFER $gid  ckpt=$ckpt"
  python infer.py \
    --test-root "$test_root" \
    --ckpt "$ckpt" \
    --out-dir "$OUT_ROOT/$gid" \
    --zip "$OUT_ROOT/${gid}.zip" \
    --no-refine --no-view-gate --no-fg-gate \
    --sigma "$SIGMA"
  ok=$((ok + 1))
done

echo "inferred $ok groups"
echo "pack with: OUT_ROOT=$OUT_ROOT PACK=$ROOT/outputs/groups_pack ZIP=$ROOT/outputs/groups_submit.zip bash scripts/pack_per_class.sh"

