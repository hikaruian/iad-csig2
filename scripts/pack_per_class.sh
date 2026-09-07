#!/usr/bin/env bash
# Concatenate per-class infer outputs into one contest zip.
# Does not call train.py / infer.py.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/per_class}"
PACK="${PACK:-$ROOT/outputs/per_class_pack}"
ZIP="${ZIP:-$ROOT/my_submission-per_class.zip}"

if [[ ! -d "$OUT_ROOT" ]]; then
  echo "no $OUT_ROOT — run infer_all_classes.sh first" >&2
  exit 1
fi

rm -rf "$PACK"
mkdir -p "$PACK/predicted_masks"
csv="$PACK/submission.csv"
printf 'group_folder,anomaly_score\n' > "$csv"

n_csv=0
n_cls=0
for cls_dir in "$OUT_ROOT"/*/; do
  [[ -d "$cls_dir" ]] || continue
  cls="$(basename "$cls_dir")"
  sub="$cls_dir/submission.csv"
  masks="$cls_dir/predicted_masks"
  if [[ ! -f "$sub" ]]; then
    echo "[warn] no submission.csv for $cls"
    continue
  fi
  # drop header, keep rows
  tail -n +2 "$sub" >> "$csv"
  rows=$(($(wc -l < "$sub") - 1))
  n_csv=$((n_csv + rows))
  if [[ -d "$masks" ]]; then
    # predicted_masks/<class>/...
    cp -a "$masks"/. "$PACK/predicted_masks/"
  else
    echo "[warn] no predicted_masks for $cls"
  fi
  n_cls=$((n_cls + 1))
  echo "packed $cls  rows=$rows"
done

# strip Windows CR if any
sed -i 's/\r$//' "$csv"

mkdir -p "$(dirname "$ZIP")"
rm -f "$ZIP"
(
  cd "$PACK"
  zip -rq "$ZIP" submission.csv predicted_masks
)

echo "classes=$n_cls  csv_rows=$n_csv"
echo "csv  $csv"
echo "zip  $ZIP"
echo "check: head $csv && unzip -l $ZIP | head"

