#!/usr/bin/env bash
# Package the stratified 1000-image fine-tuning subset (and optionally the 2000
# validation images) into one tar, check it, and optionally send it to the Jetson.
#
#   scripts/sync_subset.sh                         # check + build ade20k_subset.tar
#   scripts/sync_subset.sh --val                   # also include the 2000 validation pairs
#   scripts/sync_subset.sh --val user@nano:~/proj  # build, scp, extract and count on the Nano
#
# The image list is results/subset_train_1000.txt (made by `prune.py subset`,
# seed 0). The tar keeps the ADE20K layout (images/ and annotations/), so it
# unpacks to <dest>/ade20k-scene-parse-150/ and finetune.py / evaluate.py find it.
# For Colab: upload the tar and run `tar -xf ade20k_subset.tar -C ade20k-scene-parse-150`.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LIST="${LIST:-$REPO/results/subset_train_1000.txt}"
DATA="${DATA:-$REPO/ade20k-scene-parse-150}"
OUT="${OUT:-$REPO/ade20k_subset.tar}"
EXPECTED="${EXPECTED:-1000}"
WITH_VAL=0
DEST=""

for arg in "$@"; do
  case "$arg" in
    --val) WITH_VAL=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) DEST="$arg" ;;
  esac
done

ROOT=""
for candidate in "$DATA" "$DATA/ADEChallengeData2016"; do
  [[ -d "$candidate/images/training" && -d "$candidate/annotations/training" ]] && ROOT="$candidate" && break
done
[[ -n "$ROOT" ]] || { echo "No images/training under $DATA (set DATA=...)" >&2; exit 1; }
[[ -f "$LIST" ]] || { echo "Missing $LIST (run: python3 training/prune.py subset)" >&2; exit 1; }

# --- check the list: right count, no duplicates, every image and annotation present
count=$(grep -c . "$LIST")
dups=$(grep . "$LIST" | sort | uniq -d | wc -l)
[[ "$count" -eq "$EXPECTED" ]] || { echo "List has $count images, expected $EXPECTED" >&2; exit 1; }
[[ "$dups" -eq 0 ]] || { echo "List has $dups duplicate names" >&2; exit 1; }

FILES="$(mktemp)"; trap 'rm -f "$FILES"' EXIT
missing=0
while IFS= read -r name; do
  [[ -n "$name" ]] || continue
  stem="${name%.jpg}"
  for f in "images/training/$name" "annotations/training/$stem.png"; do
    if [[ -f "$ROOT/$f" ]]; then echo "$f" >> "$FILES"; else echo "missing: $ROOT/$f" >&2; missing=$((missing+1)); fi
  done
done < "$LIST"
[[ "$missing" -eq 0 ]] || { echo "$missing files missing; not building the tar" >&2; exit 1; }

if [[ "$WITH_VAL" -eq 1 ]]; then
  (cd "$ROOT" && find images/validation annotations/validation -type f \( -name '*.jpg' -o -name '*.png' \)) >> "$FILES"
  vals=$(grep -c '^images/validation/' "$FILES")
  [[ "$vals" -eq 2000 ]] || { echo "Found $vals validation images, expected 2000" >&2; exit 1; }
fi

tar -C "$ROOT" -cf "$OUT" -T "$FILES"
echo "List:  $LIST ($count images, sha256 $(sha256sum "$LIST" | cut -c1-12))"
echo "Built: $OUT ($(du -h "$OUT" | cut -f1), $(wc -l < "$FILES") files)"

if [[ -n "$DEST" ]]; then
  HOST="${DEST%%:*}"; DIR="${DEST#*:}"
  [[ "$HOST" != "$DEST" ]] || { echo "Destination must look like user@host:/path" >&2; exit 1; }
  scp "$OUT" "$LIST" "$HOST:/tmp/"
  # $DIR is expanded by the Jetson's shell, so ~ works.
  ssh "$HOST" "mkdir -p $DIR/ade20k-scene-parse-150 $DIR/results \
    && tar -C $DIR/ade20k-scene-parse-150 -xf /tmp/$(basename "$OUT") \
    && cp /tmp/$(basename "$LIST") $DIR/results/ \
    && echo \"On the Jetson: \$(ls $DIR/ade20k-scene-parse-150/images/training | wc -l) training images, \
\$(ls $DIR/ade20k-scene-parse-150/images/validation 2>/dev/null | wc -l) validation images\""
fi
