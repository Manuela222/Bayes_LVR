from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


CONFIGS = [
    "configs/baseline.yaml",
    "configs/extended_action.yaml",
    "configs/bayeslvr.yaml",
    "configs/bayeslvr_gated.yaml",
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
        if not any(payload["run_name"].startswith(Path(config).stem.replace(".yaml", "").replace("baseline", "ppo_baseline").replace("extended_action", "ppo_extended_action").replace("bayeslvr", "ppo_bayeslvr").replace("bayeslvr_gated", "ppo_bayeslvr_gated")) for config in CONFIGS):
            pass
        for window in payload["windows"]:
            metrics = window["metrics"]
            rows.append(
                {
                    "run_name": payload["run_name"],
                    "model": payload["run_name"].split("_seed")[0],
                    "seed": int(payload["run_name"].split("_seed")[1].split("_")[0]),
                    "window_id": window["window_id"],
                    "train_reward": metrics["train"]["cumulative_reward"],
                    "validation_reward": metrics["validation"]["cumulative_reward"],
                    "test_reward": metrics["test"]["cumulative_reward"],
                    "train_sharpe": metrics["train"]["sharpe_ratio"],
                    "validation_sharpe": metrics["validation"]["sharpe_ratio"],
                    "test_sharpe": metrics["test"]["sharpe_ratio"],
                }
            )

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby("model")
        .agg(
            mean_test_reward=("test_reward", "mean"),
            std_test_reward=("test_reward", "std"),
            mean_test_sharpe=("test_sharpe", "mean"),
            std_test_sharpe=("test_sharpe", "std"),
            mean_validation_reward=("validation_reward", "mean"),
            std_validation_reward=("validation_reward", "std"),
        )
        .reset_index()
    )
    output_prefix = ROOT / args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_prefix.with_suffix(".csv"), index=False)
    grouped.to_csv(output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".csv"), index=False)
    output_prefix.with_suffix(".json").write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".json").write_text(grouped.to_json(orient="records", indent=2), encoding="utf-8")
    print(output_prefix.with_suffix(".csv"))
    print(output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".csv"))


if __name__ == "__main__":
    main()
