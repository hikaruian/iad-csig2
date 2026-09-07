#!/usr/bin/env bash
# Split Train classes into groups of GROUP_SIZE (default 5) using symlinks.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
#TRAIN_SRC="${TRAIN_SRC:-$ROOT/data/Real-IAD/Train}"
TRAIN_SRC="${TRAIN_SRC:-/kaggle/input/datasets/dickdickgo/real-iad/Train}"
#TEST_SRC="${TEST_SRC:-$ROOT/data/Real-IAD/Test_A}"
TEST_SRC="${TEST_SRC:-/kaggle/input/datasets/dickdickgo/real-iad/Test_A}"
OUT="${GROUP_DATA:-$ROOT/data/groups}"
GROUP_SIZE="${GROUP_SIZE:-5}"

if [[ ! -d "$TRAIN_SRC" ]]; then
  echo "Set TRAIN_SRC=/path/to/Train (not found: $TRAIN_SRC)" >&2
  exit 1
fi

mapfile -t CLASSES < <(find "$TRAIN_SRC" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
n=${#CLASSES[@]}
if [[ "$n" -eq 0 ]]; then
  echo "no classes in $TRAIN_SRC" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$OUT"
list="$OUT/groups.tsv"
: > "$list"

g=0
for ((i = 0; i < n; i += GROUP_SIZE)); do
  gid="$(printf 'g%02d' "$g")"
  mkdir -p "$OUT/$gid/Train" "$OUT/$gid/Test_A"
  members=()
  for ((j = i; j < i + GROUP_SIZE && j < n; j++)); do
    cls="${CLASSES[$j]}"
    members+=("$cls")
    ln -sfn "$(cd "$TRAIN_SRC/$cls" && pwd)" "$OUT/$gid/Train/$cls"
    if [[ -d "$TEST_SRC/$cls" ]]; then
      ln -sfn "$(cd "$TEST_SRC/$cls" && pwd)" "$OUT/$gid/Test_A/$cls"
    fi
  done
  echo -e "${gid}\t${members[*]}" | tee -a "$list"
  g=$((g + 1))
done

echo "groups=$g  classes=$n  size=$GROUP_SIZE"
echo "manifest $list"
echo "data     $OUT/<gid>/{Train,Test_A}/<class>"

