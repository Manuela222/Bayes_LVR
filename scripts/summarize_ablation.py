from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="ppo_*")
    parser.add_argument("--output-prefix", default="outputs/metrics/ablation_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted((ROOT / "outputs" / "models").glob(f"{args.pattern}/summary.json")):
        payload = json.loads(summary_path.read_text())
        metrics = payload["single_run"]["metrics"]
        rows.append(
            {
                "run_name": payload["run_name"],
                "train_reward": metrics["train"]["cumulative_reward"],
                "validation_reward": metrics["validation"]["cumulative_reward"],
                "test_reward": metrics["test"]["cumulative_reward"],
                "train_sharpe": metrics["train"]["sharpe_ratio"],
                "validation_sharpe": metrics["validation"]["sharpe_ratio"],
                "test_sharpe": metrics["test"]["sharpe_ratio"],
                "train_fees": metrics["train"]["total_fees"],
                "validation_fees": metrics["validation"]["total_fees"],
                "test_fees": metrics["test"]["total_fees"],
                "train_lvr": metrics["train"]["total_lvr"],
                "validation_lvr": metrics["validation"]["total_lvr"],
                "test_lvr": metrics["test"]["total_lvr"],
                "gate_weight": metrics.get("gating", {}).get("gate_weight"),
                "gate_bias": metrics.get("gating", {}).get("gate_bias"),
                "gate_mean": metrics.get("gating", {}).get("gate_distribution", {}).get("mean"),
            }
        )

    prefix = ROOT / args.output_prefix
    ensure_dir(prefix.parent)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
