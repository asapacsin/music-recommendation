# Composite query ablation (cumulative tags in one CLAP prompt)

- **K:** 10
- **Tags:** piano → piano vocal → piano vocal relaxing (AND relevance on gold multihot)
- **No** trailing `music` in prompts
- **Fine-tuned seeds:** 5

| n_tags | query_text | n⁺ | prev | Pretrained P@K | FT P@K (mean±std) | ΔP | Pretrained ΔnDCG | FT ΔnDCG |
|--------|------------|-----|------|----------------|-------------------|-----|------------------|----------|
| 1 | piano | 29 | 0.145 | 0.200000 | 0.300000±0.000000 | 0.100000 | 0.218975 | 0.288406 |
| 2 | piano vocal | 5 | 0.025 | 0.000000 | 0.000000±0.000000 | 0.000000 | -0.042621 | -0.042621 |
| 3 | piano vocal relaxing | 4 | 0.020 | 0.000000 | 0.000000±0.000000 | 0.000000 | -0.031894 | -0.031894 |
