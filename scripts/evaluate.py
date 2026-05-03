from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.seed import configure_runtime

configure_runtime()

from stable_baselines3 import PPO

from src.bayesvol.features import BayesVolFeatureEngine
from src.envs.uniswap_env import EnvConfig, UniswapV3BayesEnv
from src.evaluation.metrics import compute_rollout_metrics
from src.utils.config import load_config
from src.utils.data import load_price_data, split_by_ratio
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="outputs/metrics/eval_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = load_price_data(config["paths"]["data"])
    splits = split_by_ratio(data, config["data"]["train_ratio"], config["data"]["validation_ratio"])
    cir_params = None
    if config["features"]["use_bayes_features"]:
        from src.trainers.ppo_trainer import fit_cir_on_train

        cir_params = fit_cir_on_train(splits.train)
    env_cfg = EnvConfig(
        delta=config["env"]["delta"],
        action_values=config["env"]["action_values"],
        x=config["env"]["initial_x"],
        gas_fee=config["env"]["gas_fee"],
        min_liquidity=config["env"]["min_liquidity"],
        use_bayes_features=config["features"]["use_bayes_features"],
    )
    bayes_engine = BayesVolFeatureEngine(cir_params) if cir_params is not None else None
    env = UniswapV3BayesEnv(splits.test, env_cfg, bayes_engine=bayes_engine)
    model = PPO.load(args.model)

    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
    metrics = compute_rollout_metrics(env.history)
    from src.trainers.ppo_trainer import compute_gating_diagnostics

    observations = np.asarray([item["observation"] for item in env.history], dtype=np.float32) if env.history else np.empty((0, env.observation_space.shape[0]), dtype=np.float32)
    metrics["gating"] = compute_gating_diagnostics(model, observations)
    write_json(metrics, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
