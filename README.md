# BayesLVR-PPO

End-to-end project for FinAI Contest 2025 Task 3, built around a simplified Uniswap v3 liquidity provision environment with Bayesian latent-volatility features for PPO.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
├── scripts/
├── src/
│   ├── bayesvol/
│   ├── data/
│   ├── envs/
│   ├── evaluation/
│   ├── models/
│   ├── trainers/
│   └── utils/
└── submission/
```

## Method

- The environment follows the main mechanics of the FinAI Uniswap v3 task: price, current tick, previous width, liquidity, EW volatility, MA24, MA168, Bollinger Bands, ADXR, BOP, and DX.
- `BayesVol` adds a discretized CIR model on latent variance, fitted on the training split only.
- The filter is a pragmatic EKF on the `r_t^2` measurement, with linearized observation `h(sigma_t^2)=sigma_t^2` and robust measurement variance `2 * sigma_t^4`.
- The Bayesian state augmentation adds these features:
  - `posterior_vol_mean`
  - `posterior_vol_var`
  - `expected_lvr_next`
  - `var_lvr_next`
  - `ci90_lvr_width`
- The Bayesian LVR proxy follows the source-project structure: `LVR ≈ c_t * sigma_t^2`, with `c_t = liquidity * sqrt(price) / (4 * portfolio_value)`.
- Optional gating applies `h_gated = h * sigmoid(w * posterior_vol_var + b)` inside the PPO feature extractor.

## Supported Variants

1. `configs/baseline.yaml`: PPO baseline.
2. `configs/extended_action.yaml`: PPO with an extended action space.
3. `configs/bayeslvr.yaml`: PPO with BayesVol features.
4. `configs/bayeslvr_gated.yaml`: PPO with BayesVol features and gating.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_data.py
```

## Usage

Quick check:

```bash
python scripts/quick_check.py
```

Single training run:

```bash
python scripts/train.py --config configs/bayeslvr_gated.yaml
```

Rolling-window training:

```bash
python scripts/train.py --config configs/bayeslvr_gated.yaml --rolling --max-windows 2
```

Model evaluation:

```bash
python scripts/evaluate.py \
  --config configs/bayeslvr_gated.yaml \
  --model outputs/models/<run_name>/model.zip \
  --output outputs/metrics/bayeslvr_gated_eval.json
```

Ablations:

```bash
python scripts/run_ablation.py
python scripts/run_ablation.py --rolling --max-windows 2
```

## Saved Artifacts

- `outputs/models/<run_name>/model.zip`
- `outputs/models/<run_name>/metrics.json`
- `outputs/models/<run_name>/config.json`
- `outputs/models/<run_name>/training_curve.png`
- `outputs/models/<run_name>/summary.json`

Metrics include at least:

- cumulative reward
- Sharpe ratio
- total fees
- total LVR
- gas cost
- fees / LVR
- action distribution
- correlation between LVR uncertainty and selected width

## Documented Assumptions

- The project uses the local dataset at `src/data/data_price_uni_h_time.csv`.
- The official starter kit and the Brini repository do not expose a separate marginal-liquidity term `l(p_t)`, so this implementation uses a consistent LVR coefficient derived from active liquidity and portfolio value.
- The environment is not rewritten from scratch; it is a compact implementation aligned with the original task logic.
- CIR parameters are fitted on squared returns from the training window and then frozen for validation and test within that same window.
- Gas costs remain fixed at 5 USD per rebalance.
