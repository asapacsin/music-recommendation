#!/bin/bash
# Deprecated alias — use scripts/sbatch_domain_tradeoff_ablation.sh
echo "NOTE: sbatch_grok_domain_tradeoff_ablation.sh is deprecated; submitting sbatch_domain_tradeoff_ablation.sh" >&2
exec sbatch "$(dirname "$0")/sbatch_domain_tradeoff_ablation.sh" "$@"
