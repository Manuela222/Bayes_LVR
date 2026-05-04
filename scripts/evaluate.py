from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.seed import configure_runtime

configure_runtime()

from stable_baselines3 import PPO

from src.envs.uniswap_env import EnvConfig, UniswapV3BayesEnv
from src.evaluation.metrics import compute_rollout_metrics
from src.utils.config import load_config
from src.utils.data import load_price_data, split_by_ratio
from src.utils.io import write_json
from src.volatility.features import build_feature_engine, feature_indices, fit_feature_engine_spec, uses_volatility_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="outputs/metrics/eval_metrics.json")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_eval_env(frame, config: dict, feature_engine_spec=None):
    env_cfg = EnvConfig(
        delta=config["env"]["delta"],
        action_values=config["env"]["action_values"],
        x=config["env"]["initial_x"],
        gas_fee=config["env"]["gas_fee"],
        min_liquidity=config["env"]["min_liquidity"],
        use_bayes_features=uses_volatility_features(config),
    )
    return UniswapV3BayesEnv(frame, env_cfg, bayes_engine=build_feature_engine(feature_engine_spec))


def rollout_with_transform(model: PPO, env: UniswapV3BayesEnv, transform=None) -> dict:
    obs, _ = env.reset()
    terminated = False
    truncated = False
    observations = []
    transformed = []
    step = 0
    while not (terminated or truncated):
        obs_array = np.asarray(obs, dtype=np.float32)
        action_obs = transform(obs_array, step) if transform is not None else obs_array
        observations.append(obs_array)
        transformed.append(np.asarray(action_obs, dtype=np.float32))
        action, _ = model.predict(action_obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
    return {
        "history": env.history,
        "observations": np.asarray(observations, dtype=np.float32),
        "transformed_observations": np.asarray(transformed, dtype=np.float32),
        "feature_names": list(env.feature_names),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = load_price_data(config["paths"]["data"])
    splits = split_by_ratio(data, config["data"]["train_ratio"], config["data"]["validation_ratio"])
    feature_engine_spec = fit_feature_engine_spec(splits.train, config)
    env = build_eval_env(splits.test, config, feature_engine_spec=feature_engine_spec)
    model = PPO.load(args.model)

    base_rollout = rollout_with_transform(model, env)
    metrics = compute_rollout_metrics(base_rollout["history"])
    from src.trainers.ppo_trainer import compute_gating_diagnostics

    observations = np.asarray([item["observation"] for item in base_rollout["history"]], dtype=np.float32) if base_rollout["history"] else np.empty((0, env.observation_space.shape[0]), dtype=np.float32)
    metrics["gating"] = compute_gating_diagnostics(model, observations)
    metrics["feature_source"] = feature_engine_spec.source if feature_engine_spec is not None else "none"

    vol_indices = feature_indices(base_rollout["feature_names"])
    rng = np.random.default_rng(args.seed)
    if vol_indices and base_rollout["observations"].shape[0] > 1:
        perm = rng.permutation(base_rollout["observations"].shape[0])

        def shuffle_transform(obs: np.ndarray, step: int) -> np.ndarray:
            modified = np.array(obs, copy=True)
            modified[vol_indices] = base_rollout["observations"][perm[min(step, len(perm) - 1)], vol_indices]
            return modified

        shuffled_rollout = rollout_with_transform(model, build_eval_env(splits.test, config, feature_engine_spec=feature_engine_spec), transform=shuffle_transform)
        metrics["shuffle_feature_eval"] = compute_rollout_metrics(shuffled_rollout["history"])

        leave_one_out = {}
        for idx in vol_indices:
            feature_name = base_rollout["feature_names"][idx]
            column_mean = float(np.mean(base_rollout["observations"][:, idx]))

            def drop_transform(obs: np.ndarray, step: int, feature_idx: int = idx, value: float = column_mean) -> np.ndarray:
                modified = np.array(obs, copy=True)
                modified[feature_idx] = value
                return modified

            dropped_rollout = rollout_with_transform(
                model,
                build_eval_env(splits.test, config, feature_engine_spec=feature_engine_spec),
                transform=drop_transform,
            )
            leave_one_out[feature_name] = compute_rollout_metrics(dropped_rollout["history"])
        metrics["leave_one_feature_out"] = leave_one_out
    write_json(metrics, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
