# Domain tradeoff report (2×2: training regime × eval domain)

Tests **forgetting vs specialization** by comparing anime-only FT (`thesis_grok_only`) vs mixed FT (`thesis_grok_mixed`: anime + MTAT + OpenMIC train, Jamendo OOD-only).

- Trade dir: `/home/mc46451/music-recommendation/data/eval/domain_tradeoff`
- Public eval root: `/home/mc46451/music-recommendation/data/eval`
- Datasets (OOD): jamendo, mtat, openmic
- Seeds: 42, 43, 44
- Top-K: 10

## 2×2 P@K (mean over seeds)

| Tag | Anime-only / Gold | Mixed / Gold | Anime-only / OOD | Mixed / OOD | Δ Gold | Δ OOD | Interpretation |
|-----|-------------------|--------------|------------------|-------------|--------|-------|----------------|
| inst_piano | 0.400 | 0.400 | 0.833 | 0.844 | +0.000 | +0.011 | No clear tradeoff — arms similar |
| inst_vocal | 0.900 | 0.800 | 0.333 | 0.600 | -0.100 | +0.267 | No clear tradeoff — arms similar |
| mood_relaxing | 0.500 | 0.500 | 0.250 | 0.367 | +0.000 | +0.117 | Forgetting-dominated (mixed recovers OOD) |

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
| inst_piano | 0.633 | 0.633 |
| inst_vocal | 0.267 | 0.267 |
| mood_relaxing | 0.200 | 0.300 |

### mtat

| Tag | anime_only | mixed |
|-----|--------------|-------|
| inst_piano | 0.867 | 0.900 |
| inst_vocal | 0.333 | 0.733 |
| mood_relaxing | 0.300 | 0.433 |

### openmic

| Tag | anime_only | mixed |
|-----|--------------|-------|
| inst_piano | 1.000 | 1.000 |
| inst_vocal | 0.400 | 0.800 |
| mood_relaxing | n/a | n/a |

## Interpretation key

| Pattern | Gold Δ (mixed − anime) | OOD Δ (mixed − anime) | Label |
|---------|------------------------|------------------------|-------|
| A | ≥ −0.05 | ≥ +0.15 (vocal) | Forgetting-dominated — mixed recovers OOD |
| B | ≥ +0.10 | ≤ −0.05 | Specialization — anime-only wins in-domain |
| C | ≤ −0.05 | ≤ −0.05 | Mixed hurts both — check ratio / leakage |
| D | ≥ −0.05 | ≥ −0.05 | No clear tradeoff — arms similar |