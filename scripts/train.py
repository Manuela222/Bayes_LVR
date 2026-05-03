from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.seed import configure_runtime, set_global_seed

configure_runtime()

from src.trainers.ppo_trainer import train_once
from src.utils.config import ROOT, load_config
from src.utils.data import build_rolling_windows, load_price_data, split_by_ratio
from src.utils.io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.timesteps is not None:
        config["ppo"]["total_timesteps"] = args.timesteps
    if args.ent_coef is not None:
        config["ppo"]["ent_coef"] = args.ent_coef
    if args.clip_range is not None:
        config["ppo"]["clip_range"] = args.clip_range
    if args.n_steps is not None:
        config["ppo"]["n_steps"] = args.n_steps
    set_global_seed(config["experiment"]["seed"])
    data = load_price_data(config["paths"]["data"])
    run_name = f"{config['experiment']['name']}_seed{config['experiment']['seed']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root_dir = ensure_dir(ROOT / config["paths"]["model_dir"] / run_name)

    summary = {"run_name": run_name, "windows": []}
    if args.rolling:
        rolling = config["rolling_window"]
        windows = build_rolling_windows(
            data,
            train_size=rolling["train_size"],
            validation_size=rolling["validation_size"],
            test_size=rolling["test_size"],
            step_size=rolling["step_size"],
        )[: args.max_windows]
        for window in windows:
            run_dir = root_dir / f"window_{window['window_id']}"
            _, artifacts, metrics = train_once(window["train"], window["validation"], window["test"], config, run_dir)
            summary["windows"].append({"window_id": window["window_id"], "artifacts": artifacts.__dict__, "metrics": metrics})
    else:
        splits = split_by_ratio(data, config["data"]["train_ratio"], config["data"]["validation_ratio"])
        _, artifacts, metrics = train_once(splits.train, splits.validation, splits.test, config, root_dir)
        summary["single_run"] = {"artifacts": artifacts.__dict__, "metrics": metrics}

    write_json(summary, root_dir / "summary.json")
    print(root_dir)


if __name__ == "__main__":
    main()
