# REVIEW — `<run_id>`

## Summary

- **Verdict**: PASS / FAIL
- **One-line reason**:

## Checklist

### Artifacts vs PLAN

- [ ] Every **expected output** from PLAN exists at the stated path
- [ ] File sizes non-zero where applicable (JSON/CSV not empty unless expected)

### Correctness

- [ ] Metric or pipeline **definition** matches intent (e.g. eval pool, K, join keys)
- [ ] No obvious logic errors in changed code (spot-check diffs)

### Reproducibility

- [ ] Another person could rerun from **PLAN.md** + **RUNLOG.md** without guessing
- [ ] Seeds / versions noted in RUNLOG if results are stochastic

### Thesis / claims safety

- [ ] Numbers cited in thesis can be traced to these output files
- [ ] Baselines (e.g. random) interpreted correctly for the documented pool

### Security

- [ ] No API keys, tokens, or private paths committed in PLAN/RUNLOG/REVIEW or code

## Defects (if FAIL)

1.
2.

## Sign-off

- Reviewer: human / Cursor
- Date:
