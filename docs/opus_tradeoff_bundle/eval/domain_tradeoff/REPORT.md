# Domain tradeoff report (2×2: training regime × eval domain)

Tests **forgetting vs specialization** by comparing anime-only FT (`thesis_tag_only`) vs mixed FT (`thesis_tag_mixed`: anime + MTAT + OpenMIC train, Jamendo OOD-only).

- Trade dir: `/home/mc46451/music-recommendation/data/eval/domain_tradeoff`
- Public eval root: `/home/mc46451/music-recommendation/data/eval`
- Datasets (OOD): jamendo, mtat, openmic
- Seeds: 42, 43, 44
- Top-K: 10

## 2×2 P@K (mean over seeds)

| Tag | Anime-only / Gold | Mixed / Gold | Anime-only / OOD | Mixed / OOD | Δ Gold | Δ OOD | Interpretation |
|-----|-------------------|--------------|------------------|-------------|--------|-------|----------------|
| inst_piano | 0.200 | 0.300 | 0.700 | 0.689 | +0.100 | -0.011 | No clear tradeoff — arms similar |
| inst_vocal | 1.000 | 0.900 | 0.367 | 0.533 | -0.100 | +0.167 | No clear tradeoff — arms similar |
| mood_relaxing | 0.500 | 0.500 | 0.283 | 0.400 | +0.000 | +0.117 | Forgetting-dominated (mixed recovers OOD) |

## Pretrained reference (OOD macro)

| Tag | Pretrained OOD |
|-----|----------------|
| inst_piano | 0.978 |
| inst_vocal | 0.756 |
| mood_relaxing | 0.533 |

## OOD per dataset (mixed arm)

### jamendo

| Tag | anime_only | mixed |
|-----|--------------|-------|
| inst_piano | 0.700 | 0.533 |
| inst_vocal | 0.300 | 0.133 |
| mood_relaxing | 0.233 | 0.467 |

### mtat

| Tag | anime_only | mixed |
|-----|--------------|-------|
| inst_piano | 0.400 | 0.533 |
| inst_vocal | 0.800 | 0.967 |
| mood_relaxing | 0.333 | 0.333 |

### openmic

| Tag | anime_only | mixed |
|-----|--------------|-------|
| inst_piano | 1.000 | 1.000 |
| inst_vocal | 0.000 | 0.500 |
| mood_relaxing | n/a | n/a |

## Interpretation key

| Pattern | Gold Δ (mixed − anime) | OOD Δ (mixed − anime) | Label |
|---------|------------------------|------------------------|-------|
| A | ≥ −0.05 | ≥ +0.15 (vocal) | Forgetting-dominated — mixed recovers OOD |
| B | ≥ +0.10 | ≤ −0.05 | Specialization — anime-only wins in-domain |
| C | ≤ −0.05 | ≤ −0.05 | Mixed hurts both — check ratio / leakage |
| D | ≥ −0.05 | ≥ −0.05 | No clear tradeoff — arms similar |