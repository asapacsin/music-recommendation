# Tag-only vs tag→LLM training ablation

Full-corpus primary-tag training text (gold join + fallback for unlabeled songs); LLM arm expands tag strings to short captions. Eval: per-checkpoint FAISS rebuild, human-gold tag queries.

- Ablation dir: `/home/mc46451/music-recommendation/data/eval/tag_llm_ablation`
- Seeds: 42, 43, 44
- Top-K: 10

## Metadata index (Grok catalog)

| Tag | Tag-only P@K | Tag→LLM P@K | Δ precision | Tag-only nDCG Δ | Tag→LLM nDCG Δ | Δ nDCG |
|-----|--------------|-------------|-------------|-----------------|----------------|--------|
| inst_piano | 0.200 | 0.200 | +0.000 | 0.150 | 0.219 | +0.069 |
| inst_vocal | 1.000 | 1.000 | +0.000 | 0.303 | 0.303 | +0.000 |
| mood_relaxing | 0.500 | 0.600 | +0.100 | 0.046 | 0.195 | +0.149 |

### Interpretation

- **inst_piano** (meta): Roughly tied (Δ precision +0.000)
- **inst_vocal** (meta): Roughly tied (Δ precision +0.000)
- **mood_relaxing** (meta): Tag→LLM arm better (Δ precision +0.100)

## Caption index (train JSONL text)

| Tag | Tag-only P@K | Tag→LLM P@K | Δ precision | Tag-only nDCG Δ | Tag→LLM nDCG Δ | Δ nDCG |
|-----|--------------|-------------|-------------|-----------------|----------------|--------|
| inst_piano | 0.400 | 0.600 | +0.200 | 0.467 | 0.594 | +0.127 |
| inst_vocal | 1.000 | 0.800 | -0.200 | 0.277 | 0.125 | -0.152 |
| mood_relaxing | 1.000 | 0.800 | -0.200 | 0.686 | 0.513 | -0.173 |

### Interpretation

- **inst_piano** (caption): Tag→LLM arm better (Δ precision +0.200)
- **inst_vocal** (caption): Tag-only arm better (Δ precision -0.200)
- **mood_relaxing** (caption): Tag-only arm better (Δ precision -0.200)
