# Full-corpus LLM vs original caption ablation

Full-corpus tag-aware LLM caption rewrite (one call per song); eval uses per-checkpoint FAISS rebuild (metadata and/or caption index).

- Ablation dir: `/home/mc46451/music-recommendation/data/eval/llm_full_ablation`
- Seeds: 42, 43, 44
- Top-K: 10

## Metadata index (Grok catalog)

| Tag | Original P@K | LLM P@K | Δ precision | Original nDCG Δ | LLM nDCG Δ | Δ nDCG |
|-----|--------------|---------|-------------|-----------------|------------|--------|
| inst_piano | 0.367 | 0.300 | -0.067 | 0.320 | 0.244 | -0.076 |
| inst_vocal | 0.967 | 0.800 | -0.167 | 0.277 | 0.166 | -0.111 |
| mood_relaxing | 0.400 | 0.400 | +0.000 | 0.106 | -0.023 | -0.129 |

### Interpretation

- **inst_piano** (meta): Original arm better (Δ precision -0.067)
- **inst_vocal** (meta): Original arm better (Δ precision -0.167)
- **mood_relaxing** (meta): Roughly tied (Δ precision +0.000)

## Caption index (train JSONL text)

| Tag | Original P@K | LLM P@K | Δ precision | Original nDCG Δ | LLM nDCG Δ | Δ nDCG |
|-----|--------------|---------|-------------|-----------------|------------|--------|
| inst_piano | 0.433 | 0.300 | -0.133 | 0.408 | 0.226 | -0.182 |
| inst_vocal | 1.000 | 0.700 | -0.300 | 0.277 | 0.019 | -0.258 |
| mood_relaxing | 0.233 | 0.300 | +0.067 | 0.019 | -0.001 | -0.020 |

### Interpretation

- **inst_piano** (caption): Original arm better (Δ precision -0.133)
- **inst_vocal** (caption): Original arm better (Δ precision -0.300)
- **mood_relaxing** (caption): LLM arm better (Δ precision +0.067)
