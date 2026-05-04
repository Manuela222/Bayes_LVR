from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.heuristics import run_heuristic_rollout
from src.trainers.ppo_trainer import build_env
from src.utils.config import load_config
from src.utils.data import build_rolling_windows, load_price_data
from src.volatility.features import fit_feature_engine_spec


CONFIGS = [
    "configs/baseline.yaml",
    "configs/extended_action.yaml",
    "configs/ewma_vol.yaml",
    "configs/garch_vol.yaml",
    "configs/bayeslvr.yaml",
    "configs/bayeslvr_attention.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--max-windows", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--output-prefix", default="outputs/metrics/robust_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item]
    allowed_prefixes = {_config_to_run_prefix(path) for path in CONFIGS}
    run_summaries = []
    for config in CONFIGS:
        for seed in seeds:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                "--config",
                str(ROOT / config),
                "--rolling",
                "--max-windows",
                str(args.max_windows),
                "--seed",
                str(seed),
                "--timesteps",
                str(args.timesteps),
                "--ent-coef",
                str(args.ent_coef),
                "--clip-range",
                str(args.clip_range),
                "--n-steps",
                str(args.n_steps),
            ]
            subprocess.run(cmd, check=True)

    rows = []
    for summary_path in sorted((ROOT / "outputs" / "models").glob("*seed*_*/summary.json")):
        payload = json.loads(summary_path.read_text())
        if not payload.get("windows"):
            continue
        run_name = payload["run_name"]
        if not any(run_name.startswith(prefix) for prefix in allowed_prefixes):
            continue
        model_name = run_name.split("_seed")[0]
        seed = int(run_name.split("_seed")[1].split("_")[0])
        for window in payload["windows"]:
            metrics = window["metrics"]
            rows.append(
                _build_row(
                    model=model_name,
                    seed=seed,
                    window_id=window["window_id"],
                    split_metrics=metrics["test"],
                    validation_metrics=metrics["validation"],
                    extra={
                        "run_name": run_name,
                        "feature_source": metrics.get("feature_source", "none"),
                    },
                )
            )

    for config_path in ["configs/garch_vol.yaml", "configs/bayeslvr.yaml"]:
        config = load_config(ROOT / config_path)
        data = load_price_data(config["paths"]["data"])
        windows = build_rolling_windows(
            data,
            train_size=config["rolling_window"]["train_size"],
            validation_size=config["rolling_window"]["validation_size"],
            test_size=config["rolling_window"]["test_size"],
            step_size=config["rolling_window"]["step_size"],
        )[: args.max_windows]
        baseline_name = "garch_heuristic" if "garch" in config_path else "cartea_style_control"
        for window in windows:
            feature_engine_spec = fit_feature_engine_spec(window["train"], config)
            test_env = build_env(window["test"], config, "test", feature_engine_spec=feature_engine_spec, bayes_feature_normalizer={})
            result = run_heuristic_rollout(test_env.unwrapped, baseline_name)
            rows.append(
                _build_row(
                    model=baseline_name,
                    seed=-1,
                    window_id=window["window_id"],
                    split_metrics=result.metrics,
                    validation_metrics={},
                    extra={"run_name": f"{baseline_name}_window{window['window_id']}", "feature_source": baseline_name},
                )
            )

    frame = pd.DataFrame(rows)
    grouped = _summarize_frame(frame)
    output_prefix = ROOT / args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_prefix.with_suffix(".csv"), index=False)
    grouped.to_csv(output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".json").write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".json").write_text(grouped.to_json(orient="records", indent=2), encoding="utf-8")
    _plot_per_run(frame, output_prefix.with_name(output_prefix.name + "_per_run_reward.png"))
    print(output_prefix.with_suffix(".csv"))
    print(output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".csv"))


def _build_row(model: str, seed: int, window_id: int, split_metrics: dict, validation_metrics: dict, extra: dict) -> dict:
    reward_boot = split_metrics.get("bootstrap", {}).get("cumulative_reward", {})
    sharpe_boot = split_metrics.get("bootstrap", {}).get("sharpe_ratio", {})
    regime = split_metrics.get("regime_performance", {})
    return {
        "run_name": extra["run_name"],
        "model": model,
        "feature_source": extra.get("feature_source", "none"),
        "seed": seed,
        "window_id": window_id,
        "validation_reward": validation_metrics.get("cumulative_reward", np.nan),
        "validation_sharpe": validation_metrics.get("sharpe_ratio", np.nan),
        "test_reward": split_metrics.get("cumulative_reward", np.nan),
        "test_reward_median": split_metrics.get("reward_median", np.nan),
        "test_reward_ci95_low": reward_boot.get("ci95_low", np.nan),
        "test_reward_ci95_high": reward_boot.get("ci95_high", np.nan),
        "test_sharpe": split_metrics.get("sharpe_ratio", np.nan),
        "test_sharpe_ci95_low": sharpe_boot.get("ci95_low", np.nan),
        "test_sharpe_ci95_high": sharpe_boot.get("ci95_high", np.nan),
        "corr_lvr_uncertainty_width": split_metrics.get("corr_lvr_uncertainty_width", np.nan),
        "low_vol_reward": regime.get("low_volatility", {}).get("cumulative_reward", np.nan),
        "high_vol_reward": regime.get("high_volatility", {}).get("cumulative_reward", np.nan),
    }


def _summarize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in frame.groupby("model"):
        rows.append(
            {
                "model": model,
                "n_runs": int(len(group)),
                "median_test_reward": float(group["test_reward"].median()),
                "mean_test_reward": float(group["test_reward"].mean()),
                "ci95_test_reward_low": float(group["test_reward"].quantile(0.025)),
                "ci95_test_reward_high": float(group["test_reward"].quantile(0.975)),
                "median_test_sharpe": float(group["test_sharpe"].median()),
                "mean_test_sharpe": float(group["test_sharpe"].mean()),
                "ci95_test_sharpe_low": float(group["test_sharpe"].quantile(0.025)),
                "ci95_test_sharpe_high": float(group["test_sharpe"].quantile(0.975)),
                "median_validation_reward": float(group["validation_reward"].median()),
                "mean_validation_reward": float(group["validation_reward"].mean()),
                "median_corr_lvr_uncertainty_width": float(group["corr_lvr_uncertainty_width"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("median_test_reward", ascending=False).reset_index(drop=True)


def _plot_per_run(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    plt.figure(figsize=(10, 5))
    for model, group in frame.groupby("model"):
        plt.scatter(group["window_id"] + 0.03 * np.arange(len(group)), group["test_reward"], label=model, alpha=0.75)
    plt.xlabel("Window id")
    plt.ylabel("Test reward")
    plt.title("Per-run test reward")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _config_to_run_prefix(config_path: str) -> str:
    stem = Path(config_path).stem
    mapping = {
        "baseline": "ppo_baseline",
        "extended_action": "ppo_extended_action",
        "bayeslvr": "ppo_bayeslvr",
        "bayeslvr_attention": "ppo_bayeslvr_attention",
        "ewma_vol": "ppo_ewma_vol",
        "garch_vol": "ppo_garch_vol",
    }
    return mapping[stem]


if __name__ == "__main__":
    main()
