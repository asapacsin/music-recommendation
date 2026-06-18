# Query-set ablation report

Expanding which **CLAP text queries** are evaluated against the metadata FAISS index (gold-labeled pool). This is **not** a checkpoint comparison table; see `summary_primary.csv` for pretrained vs fine-tuned on the same queries.

- **K:** 10
- **Pretrained matrix:** `/home/mc46451/music-recommendation/data/eval/ablation/pretrained.csv`
- **Fine-tuned matrices:** 5 seeds under `ft_seed*.csv`

## Query-set tiers (macro mean over queries in set)

| Tier | Queries | Model | Macro P@K | Macro ΔnDCG |
|------|---------|-------|-------------|---------------|
| primary_3_finetune_scope | 3 | pretrained | 0.566667 | 0.227350 |
| primary_3_finetune_scope | 3 | fine_tuned | 0.633333 | 0.288853 |
| all_style_8_multihot | 9 | pretrained | 0.222222 | 0.070168 |
| all_style_8_multihot | 9 | fine_tuned | 0.266667 | 0.128359 |
| full_style_plus_tempo_11 | 12 | pretrained | 0.250000 | 0.052550 |
| full_style_plus_tempo_11 | 12 | fine_tuned | 0.290000 | 0.113097 |

## Cumulative add-query ablation (fixed order)

Order: fine-tune **primary 3** → remaining **5 style** tags → **3 tempo** phrases.

| # added | Last query_id | Model | Macro P@K | Macro ΔnDCG |
|---------|---------------|-------|-------------|---------------|
| 1 | inst_piano | pretrained | 0.200000 | 0.218975 |
| 1 | inst_piano | fine_tuned | 0.300000 | 0.288406 |
| 2 | inst_vocal | pretrained | 0.600000 | 0.261037 |
| 2 | inst_vocal | fine_tuned | 0.650000 | 0.295752 |
| 3 | mood_relaxing | pretrained | 0.566667 | 0.227350 |
| 3 | mood_relaxing | fine_tuned | 0.633333 | 0.288853 |
| 4 | inst_orchestral | pretrained | 0.425000 | 0.162359 |
| 4 | inst_orchestral | fine_tuned | 0.475000 | 0.208486 |
| 5 | mood_sad_melancholic | pretrained | 0.340000 | 0.114217 |
| 5 | mood_sad_melancholic | fine_tuned | 0.380000 | 0.151119 |
| 6 | mood_dark_tense | pretrained | 0.283333 | 0.087219 |
| 6 | mood_dark_tense | fine_tuned | 0.333333 | 0.133153 |
| 7 | mood_exciting | pretrained | 0.242857 | 0.069036 |
| 7 | mood_exciting | fine_tuned | 0.300000 | 0.122818 |
| 8 | mood_elegant | pretrained | 0.237500 | 0.071048 |
| 8 | mood_elegant | fine_tuned | 0.300000 | 0.149886 |
| 9 | mood_epic | pretrained | 0.222222 | 0.070168 |
| 9 | mood_epic | fine_tuned | 0.266667 | 0.128359 |
| 10 | tempo_slow | pretrained | 0.200000 | 0.059269 |
| 10 | tempo_slow | fine_tuned | 0.240000 | 0.111640 |
| 11 | tempo_mid | pretrained | 0.190909 | 0.026276 |
| 11 | tempo_mid | fine_tuned | 0.243636 | 0.097645 |
| 12 | tempo_fast | pretrained | 0.250000 | 0.052550 |
| 12 | tempo_fast | fine_tuned | 0.290000 | 0.113097 |

## Per-tag results (pretrained vs fine-tuned)

| query_id | group | n+ | Pretrained P@K | FT P@K | ΔP | Pretrained ΔnDCG | FT ΔnDCG |
|----------|-------|-----|----------------|--------|-----|------------------|----------|
| inst_piano | style_primary | 29 | 0.200 | 0.300 | +0.100 | 0.219 | 0.288 |
| inst_vocal | style_primary | 139 | 1.000 | 1.000 | +0.000 | 0.303 | 0.303 |
| mood_relaxing | style_primary | 76 | 0.500 | 0.600 | +0.100 | 0.160 | 0.275 |
| inst_orchestral | style_extra | 4 | 0.000 | 0.000 | +0.000 | -0.033 | -0.033 |
| mood_sad_melancholic | style_extra | 15 | 0.000 | 0.000 | +0.000 | -0.078 | -0.078 |
| mood_dark_tense | style_extra | 6 | 0.000 | 0.100 | +0.100 | -0.048 | 0.043 |
| mood_exciting | style_extra | 6 | 0.000 | 0.100 | +0.100 | -0.040 | 0.061 |
| mood_elegant | style_extra | 15 | 0.200 | 0.300 | +0.100 | 0.085 | 0.339 |
| mood_epic | style_extra | 5 | 0.100 | 0.000 | -0.100 | 0.063 | -0.044 |
| tempo_slow | tempo | 5 | 0.000 | 0.000 | +0.000 | -0.039 | -0.039 |
| tempo_mid | tempo | 83 | 0.100 | 0.280 | +0.180 | -0.304 | -0.042 |
| tempo_fast | tempo | 112 | 0.900 | 0.800 | -0.100 | 0.342 | 0.283 |
