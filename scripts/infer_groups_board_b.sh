#!/usr/bin/env bash
# B-board: seen classes -> their 5-class group model; unseen -> unified 50-class ckpt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

#TEST_B="${TEST_B:-$ROOT/data/Real-IAD/Test_B}"
TEST_B="${TEST_B:-/kaggle/input/datasets/dickdickgo/real-iad/Test_B}"
#TEST_SEEN="${TEST_SEEN:-$ROOT/data/Real-IAD/Test_A}"
TEST_SEEN="${TEST_SEEN:-/kaggle/input/datasets/dickdickgo/real-iad/Test_B}"
GROUP_DATA="${GROUP_DATA:-$ROOT/data/groups}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/runs/groups}"
UNIFIED_CKPT="${UNIFIED_CKPT:-/kaggle/input/models/dickdickgo/best/pytorch/default/1/best.pth}"
OUT="${OUT:-$ROOT/outputs/groups_board_b}"
SIGMA="${SIGMA:-9.5}"
MANIFEST="${GROUP_DATA}/groups.tsv"

if [[ ! -d "$TEST_B" ]]; then
  echo "Set TEST_B=/path/to/Test_B" >&2
  exit 1
fi
if [[ ! -f "$UNIFIED_CKPT" ]]; then
  echo "Set UNIFIED_CKPT=.../model.pth" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "missing $MANIFEST — run make_group_links.sh on A-board Train first" >&2
  exit 1
fi

STAGE="$OUT/_parts"
rm -rf "$STAGE" "$OUT/pack"
mkdir -p "$STAGE/unseen_tree" "$OUT"

# class -> gid
declare -A CLS2GID
while IFS=$'\t' read -r gid members; do
  [[ -n "${gid:-}" ]] || continue
  for cls in $members; do
    CLS2GID["$cls"]="$gid"
  done
done < "$MANIFEST"

# Union Test_A (seen) + Test_B. Official B zip often only contains unseen folders.
declare -A CLS_SRC
while IFS= read -r cls; do
  [[ -n "$cls" ]] || continue
  CLS_SRC["$cls"]="$TEST_SEEN/$cls"
done < <(find "$TEST_SEEN" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)
while IFS= read -r cls; do
  [[ -n "$cls" ]] || continue
  CLS_SRC["$cls"]="$TEST_B/$cls"
done < <(find "$TEST_B" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

declare -A NEED_GID
unseen=0
mapfile -t ALLCLS < <(printf '%s\n' "${!CLS_SRC[@]}" | sort)
echo "classes to run: ${#ALLCLS[@]}  TEST_SEEN=$TEST_SEEN"
for cls in "${ALLCLS[@]}"; do
  src="${CLS_SRC[$cls]}"
  gid="${CLS2GID[$cls]:-}"
  if [[ -n "$gid" ]]; then
    NEED_GID["$gid"]=1
  else
    unseen=$((unseen + 1))
    ln -sfn "$(cd "$src" && pwd)" "$STAGE/unseen_tree/$cls"
    echo "[unseen] $cls  src=$src"
  fi
done

# seen: one infer per group, but Test_B may only need some classes of that group
for gid in $(printf '%s\n' "${!NEED_GID[@]}" | sort); do
  ckpt=""
  for cand in "$SAVE_ROOT/$gid/best.pth" "$SAVE_ROOT/$gid/model.pth" "$SAVE_ROOT/$gid/last.pth"; do
    [[ -f "$cand" ]] && ckpt="$cand" && break
  done
  if [[ -z "$ckpt" ]]; then
    echo "[warn] no ckpt for $gid, those classes go to unified"
    while IFS=$'\t' read -r g members; do
      [[ "$g" == "$gid" ]] || continue
      for cls in $members; do
        src="${CLS_SRC[$cls]:-}"
        [[ -n "$src" && -d "$src" ]] || continue
        ln -sfn "$(cd "$src" && pwd)" "$STAGE/unseen_tree/$cls"
      done
    done < "$MANIFEST"
    continue
  fi
  tree="$STAGE/seen_$gid"
  mkdir -p "$tree"
  while IFS=$'\t' read -r g members; do
    [[ "$g" == "$gid" ]] || continue
    for cls in $members; do
      src="${CLS_SRC[$cls]:-}"
      [[ -n "$src" && -d "$src" ]] || continue
      ln -sfn "$(cd "$src" && pwd)" "$tree/$cls"
    done
  done < "$MANIFEST"
  echo "[seen] $gid <- $ckpt"
  python infer.py \
    --test-root "$tree" \
    --ckpt "$ckpt" \
    --out-dir "$STAGE/out_$gid" \
    --zip "$STAGE/out_$gid.zip" \
    --no-refine --no-view-gate --no-fg-gate \
    --sigma "$SIGMA"
done

if [[ -n "$(find "$STAGE/unseen_tree" -mindepth 1 -maxdepth 1 -type l 2>/dev/null | head -1)" ]]; then
  echo "INFER unseen/fallback with unified"
  python infer.py \
    --test-root "$STAGE/unseen_tree" \
    --ckpt "$UNIFIED_CKPT" \
    --out-dir "$STAGE/out_unseen" \
    --zip "$STAGE/out_unseen.zip" \
    --no-refine --no-view-gate --no-fg-gate \
    --sigma "$SIGMA"
fi

PACK="$OUT/pack"
mkdir -p "$PACK/predicted_masks"
printf 'group_folder,anomaly_score\n' > "$PACK/submission.csv"
for d in "$STAGE"/out_*; do
  [[ -f "$d/submission.csv" ]] || continue
  tail -n +2 "$d/submission.csv" >> "$PACK/submission.csv"
  [[ -d "$d/predicted_masks" ]] && cp -a "$d/predicted_masks"/. "$PACK/predicted_masks/"
done
sed -i 's/\r$//' "$PACK/submission.csv"

ZIP="$OUT/submission_B_$SIGMA.zip"
rm -f "$ZIP"
( cd "$PACK" && zip -rq "$ZIP" submission.csv predicted_masks )
echo "zip $ZIP  rows=$(tail -n +2 "$PACK/submission.csv" | wc -l)"

