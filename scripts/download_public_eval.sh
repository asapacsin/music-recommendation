#!/bin/bash
# Download public eval datasets (eval-only; not wired into CLAP train by default).
#
# Usage:
#   conda activate ragweb   # needs gdown, requests, tqdm for Jamendo downloader
#   cd ~/music-recommendation && bash scripts/download_public_eval.sh
#
# Skip parts:
#   SKIP_JAMENDO=1 SKIP_MTAT=1 SKIP_OPENMIC=1 bash scripts/download_public_eval.sh
#
# Jamendo audio is large (~10GB+ for autotagging_moodtheme). OpenMIC ~2.6GB. MTAT ~2.7GB.

set -euo pipefail

REPO="${RAGWEB_REPO:-$HOME/music-recommendation}"
BASE="${PUBLIC_EVAL_DIR:-$REPO/data/public_eval}"

mkdir -p "$BASE/jamendo" "$BASE/magnatagatune" "$BASE/openmic"

JAMENDO_REPO="$BASE/jamendo/mtg-jamendo-dataset"
JAMENDO_AUDIO="$BASE/jamendo/audio"
MTAT_DIR="$BASE/magnatagatune"
OPENMIC_DIR="$BASE/openmic"
OPENMIC_TGZ="$OPENMIC_DIR/openmic-2018-v1.0.0.tgz"
OPENMIC_URL="https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz"
MTAT_BASE="https://mirg.city.ac.uk/datasets/magnatagatune"

echo "=== Public eval download → $BASE ==="

if [[ "${SKIP_JAMENDO:-0}" != "1" ]]; then
  echo "--- MTG-Jamendo (annotations + autotagging_moodtheme audio) ---"
  if [[ ! -d "$JAMENDO_REPO/.git" ]]; then
    git clone https://github.com/MTG/mtg-jamendo-dataset.git "$JAMENDO_REPO"
  else
    echo "Jamendo repo present: $JAMENDO_REPO"
  fi
  if [[ ! -f "$JAMENDO_REPO/data/autotagging_moodtheme.tsv" ]]; then
    echo "warning: expected TSV missing under $JAMENDO_REPO/data/" >&2
  fi
  mkdir -p "$JAMENDO_AUDIO"
  if find "$JAMENDO_AUDIO" -name '*.mp3' 2>/dev/null | head -1 | grep -q .; then
    echo "Jamendo audio already present under $JAMENDO_AUDIO (skipping download)"
  else
    if ! python3 -c "import gdown" 2>/dev/null; then
      echo "ERROR: Jamendo download needs gdown. Run: conda activate ragweb && pip install gdown" >&2
      exit 1
    fi
    python3 "$JAMENDO_REPO/scripts/download/download.py" \
      --dataset autotagging_moodtheme \
      --type audio \
      --from mtg-fast \
      --unpack --remove \
      "$JAMENDO_AUDIO"
  fi
else
  echo "SKIP_JAMENDO=1"
fi

if [[ "${SKIP_MTAT:-0}" != "1" ]]; then
  echo "--- MagnaTagATune (CSVs + mp3 clips) ---"
  cd "$MTAT_DIR"
  for f in annotations_final.csv clip_info_final.csv comparisons_final.csv; do
    if [[ ! -f "$f" ]]; then
      wget -c "$MTAT_BASE/$f"
    else
      echo "have $f"
    fi
  done
  if [[ ! -d mp3 ]] && [[ ! -d mp3s ]]; then
    MTAT_ZIP_001_BYTES=1100000000
    MTAT_ZIP_002_BYTES=1100000000
    MTAT_ZIP_003_BYTES=772769864
    for part in mp3.zip.001 mp3.zip.002 mp3.zip.003; do
      if [[ ! -f "$part" ]]; then
        wget -c "$MTAT_BASE/$part"
      else
        echo "have $part"
      fi
    done
    # Verify part sizes; re-fetch .003 if corrupted (common after bad cat merge)
    for spec in "mp3.zip.001:$MTAT_ZIP_001_BYTES" "mp3.zip.002:$MTAT_ZIP_002_BYTES" "mp3.zip.003:$MTAT_ZIP_003_BYTES"; do
      f="${spec%%:*}"
      want="${spec##*:}"
      got=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
      if [[ "$got" -ne "$want" ]]; then
        echo "warning: $f size $got != $want — re-downloading" >&2
        rm -f "$f"
        wget "$MTAT_BASE/$f" -O "$f"
      fi
    done
    rm -f mp3_all.zip
    echo "Joining verified split zip: cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_all.zip"
    cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3_all.zip
    unzip -t mp3_all.zip
    unzip -q mp3_all.zip
    echo "MTAT mp3 extracted"
  else
    echo "MTAT mp3 tree already present"
  fi
else
  echo "SKIP_MTAT=1"
fi

if [[ "${SKIP_OPENMIC:-0}" != "1" ]]; then
  echo "--- OpenMIC-2018 (Zenodo tarball) ---"
  cd "$OPENMIC_DIR"
  if [[ -d openmic-2018 ]] && [[ -f openmic-2018/openmic-2018-aggregated-labels.csv ]]; then
    echo "OpenMIC already extracted"
  else
    if [[ ! -f "$OPENMIC_TGZ" ]]; then
      wget -c -O "$OPENMIC_TGZ" "$OPENMIC_URL"
    fi
    tar -xzf "$OPENMIC_TGZ" -C "$OPENMIC_DIR"
  fi
  if [[ ! -d openmic-2018-tools/.git ]]; then
    git clone https://github.com/cosmir/openmic-2018.git openmic-2018-tools
  fi
else
  echo "SKIP_OPENMIC=1"
fi

echo "=== Done. Sanity checks ==="
echo -n "Jamendo mp3: "; find "$JAMENDO_AUDIO" -name '*.mp3' 2>/dev/null | wc -l || echo 0
echo -n "MTAT mp3: "; find "$MTAT_DIR" -name '*.mp3' 2>/dev/null | wc -l || echo 0
echo -n "OpenMIC ogg: "; find "$OPENMIC_DIR/openmic-2018/audio" -name '*.ogg' 2>/dev/null | wc -l || echo 0
