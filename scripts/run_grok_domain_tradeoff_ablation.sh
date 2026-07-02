#!/usr/bin/env bash
# Deprecated alias — use scripts/run_domain_tradeoff_ablation.sh
echo "NOTE: run_grok_domain_tradeoff_ablation.sh is deprecated; using run_domain_tradeoff_ablation.sh" >&2
exec bash "$(dirname "$0")/run_domain_tradeoff_ablation.sh" "$@"
