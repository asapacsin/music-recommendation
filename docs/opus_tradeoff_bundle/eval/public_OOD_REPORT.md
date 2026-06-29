# Public OOD retrieval report

- Eval root: `/home/mc46451/music-recommendation/data/eval`
- Datasets: jamendo, mtat, openmic
- Arms: pretrained, thesis_tag_only, thesis_tag_llm
- Seeds: 42, 43, 44
- Top-K: 10

## jamendo

MTG-Jamendo five-tag (split-0 test cap). Queries: piano, vocal, relaxing.

| Tag | pretrained | thesis_tag_only | thesis_tag_llm |
|-----|--------|--------|--------|
| inst_piano | 0.933 | 0.700 | 0.733 |
| inst_vocal | 0.500 | 0.300 | 0.433 |
| mood_relaxing | 0.433 | 0.233 | 0.367 |

## mtat

MagnaTagATune cap. Vocal = OR(vocals, male vocal, female voice, singer, voice). Relaxing = OR(calm, mellow).

| Tag | pretrained | thesis_tag_only | thesis_tag_llm |
|-----|--------|--------|--------|
| inst_piano | 1.000 | 0.400 | 0.967 |
| inst_vocal | 0.867 | 0.800 | 0.033 |
| mood_relaxing | 0.633 | 0.333 | 0.000 |

## openmic

OpenMIC-2018 cap. Piano + voice only (no mood labels in dataset).

| Tag | pretrained | thesis_tag_only | thesis_tag_llm |
|-----|--------|--------|--------|
| inst_piano | 1.000 | 1.000 | 1.000 |
| inst_vocal | 0.900 | 0.000 | 0.000 |
