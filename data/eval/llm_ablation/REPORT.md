# LLM vs original caption ablation

LLM arm uses gate-passed hard-clip caption replacement (~2.8% of train rows); not full-corpus LLM refinement.

- Ablation dir: `/home/mc46451/music-recommendation/data/eval/llm_ablation`
- Seeds: 42, 43, 44
- Top-K: 10
- Primary tags: piano, vocal, relaxing

| Tag | Original P@K | LLM P@K | Δ precision | Original nDCG Δ | LLM nDCG Δ | Δ nDCG |
|-----|--------------|---------|-------------|-----------------|------------|--------|
| inst_piano | 0.233 | 0.200 | -0.033 | 0.092 | 0.071 | -0.021 |
| inst_vocal | 0.700 | 0.633 | -0.067 | -0.147 | -0.199 | -0.052 |
| mood_relaxing | 0.533 | 0.600 | +0.067 | 0.096 | 0.158 | +0.062 |

## Interpretation

- **inst_piano**: Original arm better (Δ precision -0.033)
- **inst_vocal**: Original arm better (Δ precision -0.067)
- **mood_relaxing**: LLM arm better (Δ precision +0.067)
