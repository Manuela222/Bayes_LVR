from __future__ import annotations

import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.seed import configure_runtime, set_global_seed

configure_runtime()

from src.trainers.ppo_trainer import train_once
from src.utils.config import ROOT, load_config
from src.utils.data import load_price_data


def main() -> None:
    config = load_config(ROOT / "configs" / "bayeslvr_gated.yaml")
    config["ppo"]["total_timesteps"] = 64
    config["ppo"]["n_steps"] = 32
    config["ppo"]["batch_size"] = 16
    config["data"]["train_ratio"] = 0.02
    config["data"]["validation_ratio"] = 0.01
    set_global_seed(config["experiment"]["seed"])
    frame = load_price_data(config["paths"]["data"]).iloc[:900].reset_index(drop=True)
    train = frame.iloc[:600].reset_index(drop=True)
    validation = frame.iloc[600:750].reset_index(drop=True)
    test = frame.iloc[750:880].reset_index(drop=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        _, artifacts, metrics = train_once(train, validation, test, config, Path(tmp_dir))
        print("quick_check_ok")
        print(artifacts.model_path)
        print(metrics["test"]["cumulative_reward"])


if __name__ == "__main__":
    main()
