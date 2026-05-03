from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.bayesvol.cir import estimate_cir_params
from src.bayesvol.features import BayesVolFeatureEngine
from src.envs.uniswap_env import EnvConfig, UniswapV3BayesEnv
from src.evaluation.metrics import compute_rollout_metrics, summarize_gate
from src.models.feature_extractors import BayesGatedFeatureExtractor
from src.utils.io import ensure_dir, write_json
from src.utils.numeric import safe_corr, safe_log_return


class RewardTraceCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards: list[float] = []

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.rewards.append(float(reward))
        return True


@dataclass
class TrainingArtifacts:
    model_path: str
    metrics_path: str
    config_path: str
    plot_path: str


class TrainingDiagnosticsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards: list[float] = []
        self.timesteps: list[int] = []
        self.loss_history: dict[str, list[float]] = {
            "entropy_loss": [],
            "policy_gradient_loss": [],
            "value_loss": [],
            "approx_kl": [],
        }

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self.rewards.append(float(reward))
        return True

    def _on_rollout_end(self) -> None:
        self.timesteps.append(int(self.num_timesteps))
        logger_values = getattr(self.model.logger, "name_to_value", {})
        for key in self.loss_history:
            self.loss_history[key].append(float(logger_values.get(f"train/{key}", np.nan)))


def build_env(frame, config: dict, split_name: str, cir_params=None, bayes_feature_normalizer=None):
    use_bayes = bool(config["features"]["use_bayes_features"])
    bayes_engine = BayesVolFeatureEngine(cir_params) if use_bayes and cir_params is not None else None
    env_cfg = EnvConfig(
        delta=config["env"]["delta"],
        action_values=config["env"]["action_values"],
        x=config["env"]["initial_x"],
        gas_fee=config["env"]["gas_fee"],
        min_liquidity=config["env"]["min_liquidity"],
        use_bayes_features=use_bayes,
        bayes_feature_normalizer=bayes_feature_normalizer,
    )
    return Monitor(UniswapV3BayesEnv(frame, env_cfg, bayes_engine=bayes_engine), filename=None, info_keywords=("fee", "lvr", "gas_fee", "width"))


def fit_cir_on_train(train_frame) -> object:
    prices = train_frame["price"].astype(float).to_numpy()
    returns = np.array([safe_log_return(prices[i], prices[i - 1]) for i in range(1, len(prices))], dtype=float)
    return estimate_cir_params(returns, dt=1.0)


def estimate_bayes_feature_normalizer(train_frame, config: dict, cir_params) -> dict[str, dict[str, float]]:
    if not config["features"].get("normalize_bayes_features", False):
        return {}
    probe_env = build_env(train_frame, config, "normalizer_probe", cir_params=cir_params, bayes_feature_normalizer=None)
    obs, _ = probe_env.reset()
    action_values = np.asarray(config["env"]["action_values"], dtype=int)
    preferred_width = 20 if 20 in action_values else action_values[min(len(action_values) - 1, 1)]
    action_idx = int(np.where(action_values == preferred_width)[0][0])
    terminated = False
    truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = probe_env.step(action_idx)
    frame = pd.DataFrame([item["bayes"] for item in probe_env.unwrapped.history])
    normalizer = {}
    for name in BayesVolFeatureEngine.feature_names():
        if name not in frame:
            continue
        series = frame[name].astype(float)
        std = float(series.std(ddof=0))
        normalizer[name] = {"mean": float(series.mean()), "std": float(std if std > 0 else 1.0)}
    return normalizer


def train_once(train_frame, validation_frame, test_frame, config: dict, run_dir: str | Path) -> tuple[PPO, TrainingArtifacts, dict]:
    run_dir = ensure_dir(run_dir)
    cir_params = fit_cir_on_train(train_frame) if config["features"]["use_bayes_features"] else None
    bayes_feature_normalizer = estimate_bayes_feature_normalizer(train_frame, config, cir_params) if cir_params is not None else {}
    train_env = build_env(train_frame, config, "train", cir_params=cir_params, bayes_feature_normalizer=bayes_feature_normalizer)
    validation_env = build_env(validation_frame, config, "validation", cir_params=cir_params, bayes_feature_normalizer=bayes_feature_normalizer)
    test_env = build_env(test_frame, config, "test", cir_params=cir_params, bayes_feature_normalizer=bayes_feature_normalizer)

    posterior_var_index = train_env.observation_space.shape[0] - 4
    policy_kwargs = dict(
        features_extractor_class=BayesGatedFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=config["ppo"]["features_dim"],
            hidden_dim=config["ppo"]["hidden_dim"],
            activation=config["ppo"]["activation"],
            gating=bool(config["model"]["gating"]),
            posterior_var_index=posterior_var_index,
            gating_mode=config["model"].get("gating_mode", "sigmoid"),
            gate_init_std=config["model"].get("gate_init_std", 0.0),
        ),
        net_arch=dict(pi=config["ppo"]["net_arch"], vf=config["ppo"]["net_arch"]),
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        seed=config["experiment"]["seed"],
        verbose=0,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tb"),
    )
    callback = TrainingDiagnosticsCallback()
    model.learn(total_timesteps=config["ppo"]["total_timesteps"], callback=callback, progress_bar=False)

    train_rollout = rollout(model, train_env)
    validation_rollout = rollout(model, validation_env)
    test_rollout = rollout(model, test_env)
    gating_diagnostics = compute_gating_diagnostics(model, validation_rollout["observations"])
    ppo_diagnostics = {
        "timesteps": callback.timesteps,
        "entropy_loss": callback.loss_history["entropy_loss"],
        "policy_gradient_loss": callback.loss_history["policy_gradient_loss"],
        "value_loss": callback.loss_history["value_loss"],
        "approx_kl": callback.loss_history["approx_kl"],
    }
    metrics = {
        "train": compute_rollout_metrics(train_rollout["history"]),
        "validation": compute_rollout_metrics(validation_rollout["history"]),
        "test": compute_rollout_metrics(test_rollout["history"]),
        "cir_params": cir_params.to_dict() if cir_params is not None else None,
        "gating": gating_diagnostics,
        "bayes_feature_normalizer": bayes_feature_normalizer,
        "ppo_diagnostics": ppo_diagnostics,
    }

    model_path = str(run_dir / "model.zip")
    metrics_path = str(run_dir / "metrics.json")
    config_path = str(run_dir / "config.json")
    plot_path = str(run_dir / "training_curve.png")
    model.save(model_path)
    write_json(metrics, metrics_path)
    write_json(config, config_path)
    _plot_rewards(callback.rewards, plot_path)
    _plot_ppo_diagnostics(ppo_diagnostics, run_dir / "ppo_diagnostics.png")
    _save_rollout_plots(train_rollout, run_dir, "train")
    _save_rollout_plots(validation_rollout, run_dir, "validation")
    _save_rollout_plots(test_rollout, run_dir, "test")

    return model, TrainingArtifacts(model_path, metrics_path, config_path, plot_path), metrics


def rollout(model: PPO, env: Monitor) -> dict:
    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    observations = []
    while not (terminated or truncated):
        observations.append(np.asarray(obs, dtype=np.float32))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
    history = env.unwrapped.history
    return {
        "reward": total_reward,
        "history": history,
        "observations": np.asarray(observations, dtype=np.float32),
        "feature_names": list(env.unwrapped.feature_names),
    }


def compute_gating_diagnostics(model: PPO, observations: np.ndarray) -> dict:
    extractor = model.policy.features_extractor
    if not isinstance(extractor, BayesGatedFeatureExtractor):
        return {}
    result = {
        "gating_enabled": bool(extractor.gating),
        "gate_weight": float(extractor.gate_weight.detach().cpu().item()),
        "gate_bias": float(extractor.gate_bias.detach().cpu().item()),
    }
    if not extractor.gating or observations.size == 0:
        return result
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observations, device=model.device)
        posterior_var = obs_tensor[:, extractor.posterior_var_index].unsqueeze(1)
        if extractor.gating_mode == "linear":
            gates = (1.0 + extractor.gate_weight * posterior_var + extractor.gate_bias).detach().cpu().numpy().reshape(-1)
        else:
            gates = torch.sigmoid(extractor.gate_weight * posterior_var + extractor.gate_bias).detach().cpu().numpy().reshape(-1)
    result["gate_distribution"] = summarize_gate(gates)
    return result


def _plot_rewards(rewards: list[float], path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.title("Training Reward Trace")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_ppo_diagnostics(diagnostics: dict, path: Path) -> None:
    if not diagnostics["timesteps"]:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    keys = ["entropy_loss", "policy_gradient_loss", "value_loss", "approx_kl"]
    for ax, key in zip(axes.ravel(), keys):
        ax.plot(diagnostics["timesteps"], diagnostics[key])
        ax.set_title(key)
        ax.set_xlabel("Timesteps")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def _save_rollout_plots(rollout_data: dict, run_dir: Path, split_name: str) -> None:
    history = rollout_data["history"]
    if not history:
        return
    steps = np.arange(len(history))
    widths = np.asarray([item["width"] for item in history], dtype=float)
    posterior_var = np.asarray([item["bayes"]["posterior_vol_var"] for item in history], dtype=float)
    posterior_mean = np.asarray([item["bayes"]["posterior_vol_mean"] for item in history], dtype=float)
    rewards = np.asarray([item["reward"] for item in history], dtype=float)
    observations = rollout_data["observations"]
    feature_names = rollout_data["feature_names"]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, widths, lw=1)
    plt.title(f"{split_name} widths over time")
    plt.xlabel("Step")
    plt.ylabel("Width")
    plt.tight_layout()
    plt.savefig(run_dir / f"{split_name}_widths_over_time.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.scatter(posterior_var, widths, s=8, alpha=0.5)
    plt.xlabel("posterior_vol_var")
    plt.ylabel("width")
    plt.tight_layout()
    plt.savefig(run_dir / f"{split_name}_width_vs_posterior_var.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.scatter(posterior_mean, widths, s=8, alpha=0.5)
    plt.xlabel("posterior_vol_mean")
    plt.ylabel("width")
    plt.tight_layout()
    plt.savefig(run_dir / f"{split_name}_width_vs_posterior_mean.png")
    plt.close()

    bayes_feature_names = [name for name in ["posterior_vol_mean", "posterior_vol_var", "expected_lvr_next", "var_lvr_next", "ci90_lvr_width"] if name in history[0]["bayes"]]
    fig, axes = plt.subplots(1, len(bayes_feature_names), figsize=(4 * len(bayes_feature_names), 3))
    if len(bayes_feature_names) == 1:
        axes = [axes]
    for ax, feature_name in zip(axes, bayes_feature_names):
        values = np.asarray([item["bayes"][feature_name] for item in history], dtype=float)
        ax.hist(values, bins=40, alpha=0.8)
        ax.set_title(feature_name)
    plt.tight_layout()
    plt.savefig(run_dir / f"{split_name}_bayes_feature_distributions.png")
    plt.close(fig)

    scale_summary = {}
    if observations.size:
        for idx, feature_name in enumerate(feature_names):
            values = observations[:, idx]
            scale_summary[feature_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "corr_reward": float(safe_corr(values, rewards)),
                "corr_action": float(safe_corr(values, widths)),
            }
    write_json(scale_summary, run_dir / f"{split_name}_feature_scale_summary.json")
