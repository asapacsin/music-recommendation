# Gold dataset merge (human labels + program tempo + metadata)

This document records how merged evaluation data is built for **later val/test** tasks.

## File paths (defaults)

| Role | Path |
|------|------|
| Song manifest (3 clips / song) | `data/eval/song_eval_manifest.jsonl` |
| Human gold CSV (minimal) | `data/eval/gold_labels_multihot_template.csv` |
| Sidecar (join key) | `data/eval/gold_labels_multihot_template.csv.sidecar.jsonl` |
| Program tempo (BPM + CLAP song-level) | `data/eval/tempo_eval_song_predictions.jsonl` |
| LLM metadata (optional) | `data/mapping/music_metadata.json` |
| **Merged output** | `data/eval/gold_merged.jsonl` |
| Merge report | `data/eval/gold_merge_summary.json` |

## End-to-end order

1. **Build a song list** (e.g. 150 random val songs):

   ```bash
   python -m app.data_handling.music_eval_build_song_manifest \
     --filter-val-jsonl data/mapping/clap_val_15s.jsonl \
     --random-sample 150 --seed 42
   ```

2. **Generate empty human sheet** (filename + 0/1 columns; UTF-8 BOM for Excel):

   ```bash
   python -m app.data_handling.music_eval_prepare_gold_multihot_csv
   ```

3. **Label in Excel** — set each style column to `0` or `1` (`song_name` only; no path in the grid).

4. **Program tempo** — run song-level tempo eval on the **same** manifest so rows align by `source_path`:

   ```bash
   python -m app.data_handling.music_eval_zeroshot_tempo_song
   ```

   This produces `tempo_eval_song_predictions.jsonl` with BPM aggregates and CLAP zero-shot tempo per song.

5. **Merge** human + program + optional metadata:

   ```bash
   python -m app.data_handling.music_eval_merge_gold
   ```

   Options:

   - `--skip-metadata` — only human + tempo
   - `--tempo-jsonl` / `--metadata-json` / `--out` — override paths

## Gold-only manifest (tempo just for labeled rows)

When the human CSV + sidecar already exist, you do **not** need to hand-build a JSONL filter: `music_eval_build_song_manifest` reads the sidecar `source_path` lines.

1. **Manifest** (only songs in the sidecar; paths normalized; intersection with `--filter-val-jsonl` if both flags are set):

   ```bash
   python -m app.data_handling.music_eval_build_song_manifest \
     --filter-gold-sidecar data/eval/gold_labels_multihot_template.csv.sidecar.jsonl \
     --out data/eval/gold_tempo_manifest.jsonl
   ```

2. **Song-level tempo** on that manifest (see main README for `--pred-output` if you keep a separate ledger from the full-pool run).

3. **Optional coverage** before merge:

   ```bash
   python -m app.data_handling.music_eval_gold_bpm_coverage \
     --sidecar data/eval/gold_labels_multihot_template.csv.sidecar.jsonl \
     --tempo-jsonl data/eval/tempo_eval_song_predictions.jsonl
   ```

4. **Merge** as usual (`music_eval_merge_gold` picks up `--tempo-jsonl` default unless overridden).

**Review CSV (Excel):** after merge, `python -m app.data_handling.music_eval_export_gold_review_csv` writes `gold_merged_review.csv` with `song_name`, human multihot columns, and `tempo_bin_bpm` only.

If a sidecar path is missing from `music_15s_map.json`, it is omitted from the manifest; the build script prints `sidecar_paths_not_in_15s_map` in its JSON summary and a stderr warning.

## Merged record shape (one JSON per line)

Each line of `gold_merged.jsonl` contains:

- `song_name` — basename for display
- `source_path` — full relative path under project (stable key)
- `human_multihot` — your 0/1 columns (`inst_*`, `mood_*`: sad/melancholic, relaxing, dark/tense, exciting, elegant, epic)
- `program_tempo` — from tempo eval when `status=ok`:
  - `bpm_mean`, `clip_bpms`, `tempo_bin_bpm` (silver label from BPM), `tempo_clap_zeroshot`, `mean_scores_song`, etc.
- `program_metadata` — row from `music_metadata.json` matched by **basename** of `source_path` vs metadata `audio` (if found)

If tempo is missing for a song, `program_tempo` is `null` and a warning is listed in `gold_merge_summary.json`.

## Design notes

- **Taxonomy** for multihot columns is defined in **`docs/music_style.txt`**. If columns change mid-labeling, run **`music_eval_upgrade_gold_csv`** (`--in-place` creates a `.bak` backup) instead of regenerating from scratch, so row order stays aligned with the sidecar.
- **Tempo** is **not** typed in the human CSV; it is **merged from program BPM** (and CLAP head for reference).
- **Sidecar** must have **one line per CSV data row** (same order as the body rows under the header). Regenerate CSV+sidecar together if counts diverge.
- **Duplicate filenames** — rare; disambiguation uses `source_path` in the sidecar, not `song_name` alone.

## Related

- Main project README: section *Human gold set* and *Song-level tempo eval*.
