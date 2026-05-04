# NeurIPS Revision Diagnostics

## Commands

- `python scripts/run_robust_ablation.py`
- `python scripts/diagnostics_neurips_revision.py`

## Placeholder Check

- `correlations_all`: implemented
- `ekf_validation`: implemented
- `redundancy_analysis`: implemented
- `garch_heuristic`: implemented
- `cartea_baseline`: heuristic_proxy_only
- paper source present in repo root: no

## Main Results

- `ppo_baseline`: reward median 38453.000, reward 95% empirical interval [-131859.836, 61638.488], Sharpe median 13.921, Sharpe 95% empirical interval [-115.300, 20.503]
- `ppo_bayeslvr_gated`: reward median 16858.470, reward 95% empirical interval [-131859.836, 461086.613], Sharpe median 13.790, Sharpe 95% empirical interval [-115.300, 47.114]
- `ppo_bayeslvr`: reward median 16305.239, reward 95% empirical interval [-131859.836, 79644.373], Sharpe median 9.481, Sharpe 95% empirical interval [-115.300, 47.438]
- `ppo_extended_action`: reward median -64782.199, reward 95% empirical interval [-131859.836, 3911.517], Sharpe median -56.238, Sharpe 95% empirical interval [-115.300, 7.637]

## Shuffle Diagnostic

- `ppo_bayeslvr`: shuffled reward median 16305.239 vs original 16305.239, delta median 0.000; shuffled Sharpe median 9.481 vs original 9.481, delta median 0.000
- `ppo_bayeslvr_attention`: shuffled reward median 10975.775 vs original 9658.444, delta median -33.561; shuffled Sharpe median 10.988 vs original 9.945, delta median -0.180
- `ppo_bayeslvr_gated`: shuffled reward median 16733.470 vs original 16858.470, delta median 0.000; shuffled Sharpe median 13.229 vs original 13.790, delta median 0.000

## Gating

- mean gate activation 0.558905, mean corr(gate, uncertainty) -0.073661, mean corr(gate, action entropy) -0.440378, mean corr(gate, reward) 0.077766

## Claim Assessment

- Bayesian uncertainty features are informative if shuffle and leave-one-feature-out materially degrade performance or alter Sharpe.
- Standard PPO fails to exploit them robustly if the Bayes variants have unstable per-run rewards, wide intervals, or lose to simpler baselines/heuristics.
- The learned attention variant should be discussed as an alternative mechanism, not as proof that the original gating solves the robustness issue.

## Output Files

- `outputs/diagnostics_neurips_revision/main_results_with_median_ci.csv`
- `outputs/diagnostics_neurips_revision/shuffle_diagnostic.csv`
- `outputs/diagnostics_neurips_revision/leave_one_bayesvol_feature_out.csv`
- `outputs/diagnostics_neurips_revision/regime_analysis.csv`
- `outputs/diagnostics_neurips_revision/gating_diagnostics.csv`
- `outputs/diagnostics_neurips_revision/baseline_comparison.csv`
