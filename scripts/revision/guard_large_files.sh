#!/bin/bash
# Move any revision output >50MB to EffiChem_Extras_v2/revision_large/ (same relative path) and leave a symlink.
cd /export/qcai-omics/Raghvendra/PEARL
DEST=/export/qcai-omics/Raghvendra/EffiChem_Extras_v2/revision_large
for root in results/revision logs/revision scripts/revision manuscript; do
  find $root -type f -size +50M ! -type l -print0 | while IFS= read -r -d '' f; do
    mkdir -p "$DEST/$(dirname "$f")"; mv "$f" "$DEST/$f" && ln -s "$DEST/$f" "$f" && echo "moved $f"
  done
done
echo "guard ok $(date)"
