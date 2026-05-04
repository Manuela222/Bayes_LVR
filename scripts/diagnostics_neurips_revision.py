from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_rollout_metrics
from src.trainers.ppo_trainer import build_env
from src.utils.config import load_config
from src.utils.data import build_rolling_windows, load_price_data
from src.utils.io import ensure_dir, write_json
from src.utils.numeric import EPS, safe_corr
from src.volatility.features import feature_indices, fit_feature_engine_spec


OUTPUT_DIR = ROOT / "outputs" / "diagnostics_neurips_revision"
MODEL_CONFIGS = {
    "ppo_baseline": "configs/baseline.yaml",
    "ppo_extended_action": "configs/extended_action.yaml",
    "ppo_bayeslvr": "configs/bayeslvr.yaml",
    "ppo_bayeslvr_gated": "configs/bayeslvr_gated.yaml",
    "ppo_bayeslvr_attention": "configs/bayeslvr_attention.yaml",
    "ppo_ewma_vol": "configs/ewma_vol.yaml",
    "ppo_garch_vol": "configs/garch_vol.yaml",
}
CANONICAL_MODELS = ["ppo_baseline", "ppo_extended_action", "ppo_bayeslvr", "ppo_bayeslvr_gated"]
BAYES_MODELS = ["ppo_bayeslvr", "ppo_bayeslvr_gated", "ppo_bayeslvr_attention"]
COMPARISON_MODELS = [
    "ppo_baseline",
    "ppo_extended_action",
    "ppo_bayeslvr",
    "ppo_bayeslvr_gated",
    "ppo_bayeslvr_attention",
    "ppo_ewma_vol",
    "ppo_garch_vol",
]
PLACEHOLDER_TERMS = [
    "correlations_all",
    "ekf validation",
    "redundancy analysis",
    "garch heuristic",
    "cartea baseline",
    "todo",
    "fixme",
    "placeholder",
    "tbd",
]
BASELINE_FEATURE_NAMES = {
    "price",
    "tick",
    "width",
    "liquidity",
    "ew_sigma",
    "ma24",
    "ma168",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "adxr",
    "bop",
    "dx",
}


@dataclass
class RunWindow:
    model_name: str
    config_path: Path
    summary_path: Path
    run_name: str
    seed: int
    timestamp: str
    window_id: int
    model_path: Path
    metrics_path: Path
    bayes_feature_normalizer: dict


def _parse_run_name(run_name: str) -> tuple[str, int, str]:
    match = re.match(r"^(.*)_seed(\d+)_(\d{8}_\d{6})$", run_name)
    if not match:
        raise ValueError(f"Unexpected run name: {run_name}")
    return match.group(1), int(match.group(2)), match.group(3)


def _select_latest_runs(model_name: str) -> list[RunWindow]:
    summary_paths = sorted((ROOT / "outputs" / "models").glob(f"{model_name}_seed*_*/summary.json"))
    latest_by_seed: dict[int, tuple[str, Path, dict]] = {}
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        run_name = payload["run_name"]
        parsed_model, seed, timestamp = _parse_run_name(run_name)
        if parsed_model != model_name:
            continue
        current = latest_by_seed.get(seed)
        if current is None or timestamp > current[0]:
            latest_by_seed[seed] = (timestamp, summary_path, payload)

    selected: list[RunWindow] = []
    for seed in sorted(latest_by_seed):
        timestamp, summary_path, payload = latest_by_seed[seed]
        for window in payload.get("windows", []):
            window_id = int(window["window_id"])
            model_path = Path(window["artifacts"]["model_path"])
            metrics_path = Path(window["artifacts"]["metrics_path"])
            selected.append(
                RunWindow(
                    model_name=model_name,
                    config_path=ROOT / MODEL_CONFIGS[model_name],
                    summary_path=summary_path,
                    run_name=payload["run_name"],
                    seed=seed,
                    timestamp=timestamp,
                    window_id=window_id,
                    model_path=model_path,
                    metrics_path=metrics_path,
                    bayes_feature_normalizer=window["metrics"].get("bayes_feature_normalizer", {}),
                )
            )
    return selected


def collect_run_windows() -> dict[str, list[RunWindow]]:
    return {model_name: _select_latest_runs(model_name) for model_name in MODEL_CONFIGS}


def placeholder_scan() -> tuple[pd.DataFrame, dict]:
    allowed_suffixes = {".py", ".md", ".tex", ".yaml", ".yml", ".json"}
    search_roots = [ROOT / "src", ROOT / "scripts", ROOT / "configs", ROOT / "submission", ROOT / "README.md"]
    excluded_paths = {ROOT / "scripts" / "diagnostics_neurips_revision.py"}
    rows = []
    found_terms: dict[str, list[dict[str, object]]] = defaultdict(list)
    for root in search_roots:
        paths = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed_suffixes)
        for path in paths:
            if path in excluded_paths:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            lower = text.lower()
            for term in PLACEHOLDER_TERMS:
                if term in lower:
                    for lineno, line in enumerate(text.splitlines(), start=1):
                        if term in line.lower():
                            rows.append({"term": term, "path": str(path.relative_to(ROOT)), "line": lineno, "text": line.strip()[:240]})
                            found_terms[term].append({"path": str(path.relative_to(ROOT)), "line": lineno, "text": line.strip()[:240]})
    status = {
        "correlations_all": "implemented" if found_terms.get("correlations_all") else "missing",
        "ekf_validation": "implemented" if any("ekf validation" in item["text"].lower() or item["path"].endswith("metrics.py") for item in found_terms.get("ekf validation", [])) or (ROOT / "src" / "evaluation" / "metrics.py").read_text(encoding="utf-8").lower().find("ekf_validation") >= 0 else "missing",
        "redundancy_analysis": "implemented" if (ROOT / "src" / "evaluation" / "metrics.py").read_text(encoding="utf-8").lower().find("redundancy_analysis") >= 0 else "missing",
        "garch_heuristic": "implemented" if (ROOT / "src" / "baselines" / "heuristics.py").read_text(encoding="utf-8").lower().find("garch_heuristic") >= 0 else "missing",
        "cartea_baseline": "heuristic_proxy_only" if (ROOT / "src" / "baselines" / "heuristics.py").read_text(encoding="utf-8").lower().find("cartea_style_control") >= 0 else "missing",
        "paper_source_present": "yes" if any(path.suffix.lower() == ".tex" for path in ROOT.glob("*.tex")) else "no",
    }
    frame = pd.DataFrame(rows).sort_values(["term", "path", "line"]).reset_index(drop=True) if rows else pd.DataFrame(columns=["term", "path", "line", "text"])
    return frame, status


def _load_window_frames(config_path: Path, window_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    data = load_price_data(config["paths"]["data"])
    windows = build_rolling_windows(
        data,
        train_size=config["rolling_window"]["train_size"],
        validation_size=config["rolling_window"]["validation_size"],
        test_size=config["rolling_window"]["test_size"],
        step_size=config["rolling_window"]["step_size"],
    )
    selected = windows[window_id]
    return selected["train"], selected["validation"], selected["test"]


def _rollout_model(model: PPO, env, transform=None, feature_names_override: list[str] | None = None) -> dict:
    obs, _ = env.reset()
    terminated = False
    truncated = False
    original_observations = []
    policy_observations = []
    action_probs = []
    action_entropy = []
    step = 0
    while not (terminated or truncated):
        raw_obs = np.asarray(obs, dtype=np.float32)
        policy_obs = np.asarray(transform(raw_obs, step) if transform is not None else raw_obs, dtype=np.float32)
        obs_tensor = torch.as_tensor(policy_obs[None, :], device=model.device)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        entropy = float(dist.entropy().detach().cpu().numpy()[0])
        action, _ = model.predict(policy_obs, deterministic=True)
        original_observations.append(raw_obs)
        policy_observations.append(policy_obs)
        action_probs.append(probs)
        action_entropy.append(entropy)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
    history = env.unwrapped.history
    metrics = compute_rollout_metrics(history)
    return {
        "metrics": metrics,
        "history": history,
        "observations": np.asarray(original_observations, dtype=np.float32),
        "policy_observations": np.asarray(policy_observations, dtype=np.float32),
        "action_probs": np.asarray(action_probs, dtype=np.float32),
        "action_entropy": np.asarray(action_entropy, dtype=np.float32),
        "feature_names": feature_names_override or list(env.unwrapped.feature_names),
    }


def _build_eval_env(run: RunWindow):
    train_frame, _, test_frame = _load_window_frames(run.config_path, run.window_id)
    config = load_config(run.config_path)
    feature_engine_spec = fit_feature_engine_spec(train_frame, config)
    env = build_env(
        test_frame,
        config,
        split_name=f"test_window_{run.window_id}",
        feature_engine_spec=feature_engine_spec,
        bayes_feature_normalizer=run.bayes_feature_normalizer,
    )
    return env, train_frame, test_frame


def evaluate_run(run: RunWindow) -> dict:
    env, _, _ = _build_eval_env(run)
    model = PPO.load(str(run.model_path))
    result = _rollout_model(model, env)
    result["run"] = run
    return result


def _aggregate_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _bootstrap_sample(values: np.ndarray, n_boot: int = 10000, seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = values.astype(float)
    draws = []
    for _ in range(n_boot):
        picked = samples[rng.integers(0, samples.size, size=samples.size)]
        draws.append(float(np.mean(picked)))
    draws_arr = np.asarray(draws, dtype=float)
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples, ddof=0)),
        "ci95_low": float(np.quantile(draws_arr, 0.025)),
        "ci95_high": float(np.quantile(draws_arr, 0.975)),
    }


def _regime_rows(model_name: str, seed: int, window_id: int, history: list[dict]) -> list[dict]:
    sigma = np.asarray([item.get("sigma_empirical", item["bayes"].get("empirical_vol_proxy", 0.0)) for item in history], dtype=float)
    widths = np.asarray([item["width"] for item in history], dtype=float)
    actions = np.asarray([item["action"] for item in history], dtype=float)
    rewards = np.asarray([item["reward"] for item in history], dtype=float)
    quantile_specs = {"median": [0.5], "quartile": [0.25, 0.5, 0.75]}
    rows = []
    for split_name, cuts in quantile_specs.items():
        boundaries = np.quantile(sigma, cuts)
        if split_name == "median":
            masks = {
                "low": sigma <= boundaries[0],
                "high": sigma > boundaries[0],
            }
        else:
            masks = {
                "q1": sigma <= boundaries[0],
                "q2": (sigma > boundaries[0]) & (sigma <= boundaries[1]),
                "q3": (sigma > boundaries[1]) & (sigma <= boundaries[2]),
                "q4": sigma > boundaries[2],
            }
        for regime_label, mask in masks.items():
            if not np.any(mask):
                continue
            regime_rewards = rewards[mask]
            regime_widths = widths[mask]
            regime_actions = actions[mask]
            if regime_rewards.size > 1 and float(np.std(regime_rewards)) > EPS:
                sharpe = float(np.mean(regime_rewards) / np.std(regime_rewards) * np.sqrt(24.0 * 365.0))
            else:
                sharpe = 0.0
            action_distribution = {
                str(int(action)): int(count)
                for action, count in pd.Series(regime_actions).value_counts().sort_index().items()
            }
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "window_id": window_id,
                    "split_type": split_name,
                    "regime": regime_label,
                    "n_steps": int(mask.sum()),
                    "sigma_low": float(np.min(sigma[mask])),
                    "sigma_high": float(np.max(sigma[mask])),
                    "cumulative_reward": float(np.sum(regime_rewards)),
                    "avg_step_reward": float(np.mean(regime_rewards)),
                    "median_step_reward": float(np.median(regime_rewards)),
                    "sharpe_ratio": sharpe,
                    "width_mean": float(np.mean(regime_widths)),
                    "width_median": float(np.median(regime_widths)),
                    "width_std": float(np.std(regime_widths, ddof=0)),
                    "action_distribution": json.dumps(action_distribution, sort_keys=True),
                }
            )
    return rows


def _shuffle_and_ablation_rows(run_eval: dict) -> tuple[list[dict], list[dict]]:
    run: RunWindow = run_eval["run"]
    feature_names = run_eval["feature_names"]
    observations = run_eval["observations"]
    bayes_cols = [idx for idx, name in enumerate(feature_names) if name not in BASELINE_FEATURE_NAMES]
    if not bayes_cols:
        return [], []
    rng = np.random.default_rng(run.seed * 100 + run.window_id)
    permutation = rng.permutation(observations.shape[0])
    base_reward = float(run_eval["metrics"]["cumulative_reward"])
    base_sharpe = float(run_eval["metrics"]["sharpe_ratio"])
    model = PPO.load(str(run.model_path))

    env_shuffle, _, _ = _build_eval_env(run)

    def shuffle_transform(obs: np.ndarray, step: int) -> np.ndarray:
        transformed = np.array(obs, copy=True)
        transformed[bayes_cols] = observations[permutation[min(step, len(permutation) - 1)], bayes_cols]
        return transformed

    shuffled = _rollout_model(model, env_shuffle, transform=shuffle_transform, feature_names_override=feature_names)
    shuffle_rows = [
        {
            "model": run.model_name,
            "seed": run.seed,
            "window_id": run.window_id,
            "original_reward": base_reward,
            "shuffled_reward": float(shuffled["metrics"]["cumulative_reward"]),
            "delta_reward": float(shuffled["metrics"]["cumulative_reward"] - base_reward),
            "original_sharpe": base_sharpe,
            "shuffled_sharpe": float(shuffled["metrics"]["sharpe_ratio"]),
            "delta_sharpe": float(shuffled["metrics"]["sharpe_ratio"] - base_sharpe),
        }
    ]

    ablation_rows = []
    for col in bayes_cols:
        feature_name = feature_names[col]
        fill_value = float(np.mean(observations[:, col]))

        def drop_transform(obs: np.ndarray, step: int, feature_idx: int = col, replacement: float = fill_value) -> np.ndarray:
            transformed = np.array(obs, copy=True)
            transformed[feature_idx] = replacement
            return transformed

        env_drop, _, _ = _build_eval_env(run)
        dropped = _rollout_model(model, env_drop, transform=drop_transform, feature_names_override=feature_names)
        ablation_rows.append(
            {
                "model": run.model_name,
                "seed": run.seed,
                "window_id": run.window_id,
                "dropped_feature": feature_name,
                "original_reward": base_reward,
                "ablated_reward": float(dropped["metrics"]["cumulative_reward"]),
                "delta_reward": float(dropped["metrics"]["cumulative_reward"] - base_reward),
                "original_sharpe": base_sharpe,
                "ablated_sharpe": float(dropped["metrics"]["sharpe_ratio"]),
                "delta_sharpe": float(dropped["metrics"]["sharpe_ratio"] - base_sharpe),
            }
        )
    return shuffle_rows, ablation_rows


def _gate_rows(run_eval: dict) -> list[dict]:
    run: RunWindow = run_eval["run"]
    if run.model_name != "ppo_bayeslvr_gated":
        return []
    model = PPO.load(str(run.model_path))
    extractor = model.policy.features_extractor
    observations = run_eval["observations"]
    rewards = np.asarray([item["reward"] for item in run_eval["history"]], dtype=float)
    uncertainty = np.asarray([item["bayes"]["posterior_vol_var"] for item in run_eval["history"]], dtype=float)
    action_entropy = run_eval["action_entropy"].astype(float)
    obs_tensor = torch.as_tensor(observations, device=model.device)
    posterior_var = obs_tensor[:, extractor.posterior_var_index].unsqueeze(1)
    with torch.no_grad():
        if extractor.gating_mode == "linear":
            gates = (1.0 + extractor.gate_weight * posterior_var + extractor.gate_bias).detach().cpu().numpy().reshape(-1)
        else:
            gates = torch.sigmoid(extractor.gate_weight * posterior_var + extractor.gate_bias).detach().cpu().numpy().reshape(-1)
    return [
        {
            "model": run.model_name,
            "seed": run.seed,
            "window_id": run.window_id,
            "gate_mean": float(np.mean(gates)),
            "gate_std": float(np.std(gates, ddof=0)),
            "gate_min": float(np.min(gates)),
            "gate_median": float(np.median(gates)),
            "gate_max": float(np.max(gates)),
            "corr_gate_uncertainty": float(safe_corr(gates, uncertainty)),
            "corr_gate_action_entropy": float(safe_corr(gates, action_entropy)),
            "corr_gate_reward": float(safe_corr(gates, rewards)),
            "gate_weight": float(extractor.gate_weight.detach().cpu().item()),
            "gate_bias": float(extractor.gate_bias.detach().cpu().item()),
        }
    ]


def _collect_per_run_rows(run_eval: dict) -> dict:
    run: RunWindow = run_eval["run"]
    metrics = run_eval["metrics"]
    reward_boot = metrics["bootstrap"]["cumulative_reward"]
    sharpe_boot = metrics["bootstrap"]["sharpe_ratio"]
    return {
        "model": run.model_name,
        "seed": run.seed,
        "window_id": run.window_id,
        "run_name": run.run_name,
        "cumulative_reward": float(metrics["cumulative_reward"]),
        "reward_median": float(metrics["reward_median"]),
        "reward_ci95_low": float(reward_boot["ci95_low"]),
        "reward_ci95_high": float(reward_boot["ci95_high"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"]),
        "sharpe_ci95_low": float(sharpe_boot["ci95_low"]),
        "sharpe_ci95_high": float(sharpe_boot["ci95_high"]),
        "total_fees": float(metrics["total_fees"]),
        "total_lvr": float(metrics["total_lvr"]),
        "gas_cost": float(metrics["gas_cost"]),
        "fees_to_lvr_ratio": float(metrics["fees_to_lvr_ratio"]),
        "corr_lvr_uncertainty_width": float(metrics["corr_lvr_uncertainty_width"]),
    }


def _main_results_table(per_run_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in CANONICAL_MODELS:
        subset = per_run_frame[per_run_frame["model"] == model_name]
        rows.append(
            {
                "model": model_name,
                "n_runs": int(len(subset)),
                "reward_mean": float(subset["cumulative_reward"].mean()),
                "reward_median": float(subset["cumulative_reward"].median()),
                "reward_std": float(subset["cumulative_reward"].std(ddof=0)),
                "reward_ci95_low": float(np.quantile(subset["cumulative_reward"], 0.025)),
                "reward_ci95_high": float(np.quantile(subset["cumulative_reward"], 0.975)),
                "sharpe_mean": float(subset["sharpe_ratio"].mean()),
                "sharpe_median": float(subset["sharpe_ratio"].median()),
                "sharpe_std": float(subset["sharpe_ratio"].std(ddof=0)),
                "sharpe_ci95_low": float(np.quantile(subset["sharpe_ratio"], 0.025)),
                "sharpe_ci95_high": float(np.quantile(subset["sharpe_ratio"], 0.975)),
                "reward_bootstrap_mean": _bootstrap_sample(subset["cumulative_reward"].to_numpy())["mean"],
                "reward_bootstrap_ci95_low": _bootstrap_sample(subset["cumulative_reward"].to_numpy())["ci95_low"],
                "reward_bootstrap_ci95_high": _bootstrap_sample(subset["cumulative_reward"].to_numpy())["ci95_high"],
                "sharpe_bootstrap_mean": _bootstrap_sample(subset["sharpe_ratio"].to_numpy())["mean"],
                "sharpe_bootstrap_ci95_low": _bootstrap_sample(subset["sharpe_ratio"].to_numpy())["ci95_low"],
                "sharpe_bootstrap_ci95_high": _bootstrap_sample(subset["sharpe_ratio"].to_numpy())["ci95_high"],
            }
        )
    return pd.DataFrame(rows).sort_values("reward_median", ascending=False).reset_index(drop=True)


def _baseline_comparison_table(per_run_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in COMPARISON_MODELS:
        subset = per_run_frame[per_run_frame["model"] == model_name]
        rows.append(
            {
                "model": model_name,
                "type": "ppo",
                "n_runs": int(len(subset)),
                "reward_mean": float(subset["cumulative_reward"].mean()),
                "reward_median": float(subset["cumulative_reward"].median()),
                "reward_std": float(subset["cumulative_reward"].std(ddof=0)),
                "sharpe_mean": float(subset["sharpe_ratio"].mean()),
                "sharpe_median": float(subset["sharpe_ratio"].median()),
                "sharpe_std": float(subset["sharpe_ratio"].std(ddof=0)),
            }
        )
    heuristic_frame = pd.read_csv(ROOT / "outputs" / "metrics" / "robust_ablation.csv")
    heuristic_subset = heuristic_frame[heuristic_frame["model"].isin(["garch_heuristic", "cartea_style_control"])]
    for model_name, group in heuristic_subset.groupby("model"):
        rows.append(
            {
                "model": model_name,
                "type": "heuristic",
                "n_runs": int(len(group)),
                "reward_mean": float(group["test_reward"].mean()),
                "reward_median": float(group["test_reward"].median()),
                "reward_std": float(group["test_reward"].std(ddof=0)),
                "sharpe_mean": float(group["test_sharpe"].mean()),
                "sharpe_median": float(group["test_sharpe"].median()),
                "sharpe_std": float(group["test_sharpe"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("reward_median", ascending=False).reset_index(drop=True)


def _write_table(frame: pd.DataFrame, stem: str) -> None:
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    json_path = OUTPUT_DIR / f"{stem}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")


def _plot_per_run_reward(per_run_frame: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 5))
    for model_name, group in per_run_frame.groupby("model"):
        x = np.arange(len(group))
        labels = [f"{seed}-w{window}" for seed, window in zip(group["seed"], group["window_id"])]
        plt.scatter(x, group["cumulative_reward"], label=model_name, alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.ylabel("Cumulative reward")
    plt.title("Per-run reward by model")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_run_reward_by_model.png")
    plt.close()


def _plot_reward_distribution(per_run_frame: pd.DataFrame) -> None:
    order = list(per_run_frame.groupby("model")["cumulative_reward"].median().sort_values(ascending=False).index)
    data = [per_run_frame.loc[per_run_frame["model"] == model_name, "cumulative_reward"].to_numpy() for model_name in order]
    plt.figure(figsize=(11, 5))
    plt.boxplot(data, labels=order, vert=True)
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Cumulative reward")
    plt.title("Reward distribution by model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reward_distribution_by_model.png")
    plt.close()


def _plot_shuffle_vs_original(shuffle_frame: pd.DataFrame) -> None:
    summary = (
        shuffle_frame.groupby("model")
        .agg(original_reward=("original_reward", "mean"), shuffled_reward=("shuffled_reward", "mean"))
        .reset_index()
    )
    x = np.arange(len(summary))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, summary["original_reward"], width, label="original")
    plt.bar(x + width / 2, summary["shuffled_reward"], width, label="shuffled")
    plt.xticks(x, summary["model"], rotation=20, ha="right")
    plt.ylabel("Mean cumulative reward")
    plt.title("BayesVol shuffle diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bayesvol_shuffle_vs_original.png")
    plt.close()


def _plot_regime_performance(regime_frame: pd.DataFrame) -> None:
    median_frame = regime_frame[regime_frame["split_type"] == "median"].copy()
    summary = (
        median_frame.groupby(["model", "regime"])["avg_step_reward"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(COMPARISON_MODELS)
    )
    summary.plot(kind="bar", figsize=(11, 5))
    plt.ylabel("Average step reward")
    plt.title("High-volatility vs low-volatility regime performance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "high_vs_low_volatility_regime_performance.png")
    plt.close()


def _plot_action_distribution_by_uncertainty(regime_frame: pd.DataFrame) -> None:
    subset = regime_frame[(regime_frame["model"] == "ppo_bayeslvr") & (regime_frame["split_type"] == "median")]
    action_labels = sorted(
        {
            action
            for payload in subset["action_distribution"]
            for action in json.loads(payload).keys()
        },
        key=int,
    )
    positions = np.arange(len(action_labels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    for idx, regime_label in enumerate(["low", "high"]):
        group = subset[subset["regime"] == regime_label]
        counts = defaultdict(int)
        for payload in group["action_distribution"]:
            for action, count in json.loads(payload).items():
                counts[action] += int(count)
        values = [counts[action] for action in action_labels]
        plt.bar(positions + (idx - 0.5) * width, values, width=width, label=regime_label)
    plt.xticks(positions, action_labels)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action distribution by uncertainty regime (ppo_bayeslvr)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "action_distribution_by_uncertainty_regime.png")
    plt.close()


def _plot_gate_vs_posterior_variance(gating_frame: pd.DataFrame, gating_points: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(gating_points["posterior_var"], gating_points["gate"], s=6, alpha=0.35)
    plt.xlabel("posterior_vol_var")
    plt.ylabel("gate activation")
    plt.title("Gate activation vs posterior variance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gate_activation_vs_posterior_variance.png")
    plt.close()


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    placeholder_frame, placeholder_status = placeholder_scan()
    _write_table(placeholder_frame, "placeholder_scan")
    write_json(placeholder_status, OUTPUT_DIR / "placeholder_status.json")

    run_windows = collect_run_windows()

    all_original_rows = []
    regime_rows = []
    shuffle_rows = []
    ablation_rows = []
    gating_rows = []
    gate_points = []

    for model_name, runs in run_windows.items():
        for run in runs:
            evaluation = evaluate_run(run)
            all_original_rows.append(_collect_per_run_rows(evaluation))
            regime_rows.extend(_regime_rows(run.model_name, run.seed, run.window_id, evaluation["history"]))
            if model_name in BAYES_MODELS:
                current_shuffle_rows, current_ablation_rows = _shuffle_and_ablation_rows(evaluation)
                shuffle_rows.extend(current_shuffle_rows)
                ablation_rows.extend(current_ablation_rows)
            if model_name == "ppo_bayeslvr_gated":
                gating_rows.extend(_gate_rows(evaluation))
                model = PPO.load(str(run.model_path))
                extractor = model.policy.features_extractor
                obs_tensor = torch.as_tensor(evaluation["observations"], device=model.device)
                posterior_var = obs_tensor[:, extractor.posterior_var_index].unsqueeze(1)
                with torch.no_grad():
                    if extractor.gating_mode == "linear":
                        gates = (1.0 + extractor.gate_weight * posterior_var + extractor.gate_bias).detach().cpu().numpy().reshape(-1)
                    else:
                        gates = torch.sigmoid(extractor.gate_weight * posterior_var + extractor.gate_bias).detach().cpu().numpy().reshape(-1)
                gate_points.extend(
                    {
                        "seed": run.seed,
                        "window_id": run.window_id,
                        "posterior_var": float(pv),
                        "gate": float(gate),
                    }
                    for pv, gate in zip(np.asarray([item["bayes"]["posterior_vol_var"] for item in evaluation["history"]], dtype=float), gates)
                )

    per_run_frame = pd.DataFrame(all_original_rows).sort_values(["model", "seed", "window_id"]).reset_index(drop=True)
    regime_frame = pd.DataFrame(regime_rows).sort_values(["model", "seed", "window_id", "split_type", "regime"]).reset_index(drop=True)
    shuffle_frame = pd.DataFrame(shuffle_rows).sort_values(["model", "seed", "window_id"]).reset_index(drop=True)
    ablation_frame = pd.DataFrame(ablation_rows).sort_values(["model", "dropped_feature", "seed", "window_id"]).reset_index(drop=True)
    gating_frame = pd.DataFrame(gating_rows).sort_values(["seed", "window_id"]).reset_index(drop=True)
    gate_points_frame = pd.DataFrame(gate_points)

    if not shuffle_frame.empty:
        shuffle_agg = (
            shuffle_frame.groupby("model")
            .agg(
                original_reward_mean=("original_reward", "mean"),
                original_reward_std=("original_reward", "std"),
                original_reward_median=("original_reward", "median"),
                shuffled_reward_mean=("shuffled_reward", "mean"),
                shuffled_reward_std=("shuffled_reward", "std"),
                shuffled_reward_median=("shuffled_reward", "median"),
                delta_reward_mean=("delta_reward", "mean"),
                delta_reward_std=("delta_reward", "std"),
                delta_reward_median=("delta_reward", "median"),
                original_sharpe_mean=("original_sharpe", "mean"),
                original_sharpe_std=("original_sharpe", "std"),
                original_sharpe_median=("original_sharpe", "median"),
                shuffled_sharpe_mean=("shuffled_sharpe", "mean"),
                shuffled_sharpe_std=("shuffled_sharpe", "std"),
                shuffled_sharpe_median=("shuffled_sharpe", "median"),
                delta_sharpe_mean=("delta_sharpe", "mean"),
                delta_sharpe_std=("delta_sharpe", "std"),
                delta_sharpe_median=("delta_sharpe", "median"),
            )
            .reset_index()
        )
    else:
        shuffle_agg = pd.DataFrame()

    if not ablation_frame.empty:
        ablation_agg = (
            ablation_frame.groupby(["model", "dropped_feature"])
            .agg(
                ablated_reward_mean=("ablated_reward", "mean"),
                ablated_reward_std=("ablated_reward", "std"),
                ablated_reward_median=("ablated_reward", "median"),
                delta_reward_mean=("delta_reward", "mean"),
                delta_reward_std=("delta_reward", "std"),
                delta_reward_median=("delta_reward", "median"),
                ablated_sharpe_mean=("ablated_sharpe", "mean"),
                ablated_sharpe_std=("ablated_sharpe", "std"),
                ablated_sharpe_median=("ablated_sharpe", "median"),
                delta_sharpe_mean=("delta_sharpe", "mean"),
                delta_sharpe_std=("delta_sharpe", "std"),
                delta_sharpe_median=("delta_sharpe", "median"),
            )
            .reset_index()
        )
    else:
        ablation_agg = pd.DataFrame()

    main_results = _main_results_table(per_run_frame)
    baseline_comparison = _baseline_comparison_table(per_run_frame)

    _write_table(per_run_frame, "per_run_original_evaluation")
    _write_table(main_results, "main_results_with_median_ci")
    _write_table(shuffle_frame, "shuffle_diagnostic")
    _write_table(shuffle_agg, "shuffle_diagnostic_summary")
    _write_table(ablation_frame, "leave_one_bayesvol_feature_out")
    _write_table(ablation_agg, "leave_one_bayesvol_feature_out_summary")
    _write_table(regime_frame, "regime_analysis")
    _write_table(gating_frame, "gating_diagnostics")
    _write_table(baseline_comparison, "baseline_comparison")
    if not gate_points_frame.empty:
        _write_table(gate_points_frame, "gating_points")

    _plot_per_run_reward(per_run_frame[per_run_frame["model"].isin(COMPARISON_MODELS)])
    _plot_reward_distribution(per_run_frame[per_run_frame["model"].isin(COMPARISON_MODELS)])
    if not shuffle_frame.empty:
        _plot_shuffle_vs_original(shuffle_frame)
    _plot_regime_performance(regime_frame)
    _plot_action_distribution_by_uncertainty(regime_frame)
    if not gate_points_frame.empty:
        _plot_gate_vs_posterior_variance(gating_frame, gate_points_frame)

    report_path = OUTPUT_DIR / "neurips_revision_report.md"
    report_lines = [
        "# NeurIPS Revision Diagnostics",
        "",
        "## Commands",
        "",
        "- `python scripts/run_robust_ablation.py`",
        "- `python scripts/diagnostics_neurips_revision.py`",
        "",
        "## Placeholder Check",
        "",
        f"- `correlations_all`: {placeholder_status['correlations_all']}",
        f"- `ekf_validation`: {placeholder_status['ekf_validation']}",
        f"- `redundancy_analysis`: {placeholder_status['redundancy_analysis']}",
        f"- `garch_heuristic`: {placeholder_status['garch_heuristic']}",
        f"- `cartea_baseline`: {placeholder_status['cartea_baseline']}",
        f"- paper source present in repo root: {placeholder_status['paper_source_present']}",
        "",
        "## Main Results",
        "",
    ]
    for row in main_results.to_dict(orient="records"):
        report_lines.append(
            f"- `{row['model']}`: reward median {row['reward_median']:.3f}, reward 95% empirical interval [{row['reward_ci95_low']:.3f}, {row['reward_ci95_high']:.3f}], "
            f"Sharpe median {row['sharpe_median']:.3f}, Sharpe 95% empirical interval [{row['sharpe_ci95_low']:.3f}, {row['sharpe_ci95_high']:.3f}]"
        )
    report_lines.extend(
        [
            "",
            "## Shuffle Diagnostic",
            "",
        ]
    )
    for row in shuffle_agg.to_dict(orient="records"):
        report_lines.append(
            f"- `{row['model']}`: shuffled reward median {row['shuffled_reward_median']:.3f} vs original {row['original_reward_median']:.3f}, delta median {row['delta_reward_median']:.3f}; "
            f"shuffled Sharpe median {row['shuffled_sharpe_median']:.3f} vs original {row['original_sharpe_median']:.3f}, delta median {row['delta_sharpe_median']:.3f}"
        )
    report_lines.extend(
        [
            "",
            "## Gating",
            "",
        ]
    )
    if gating_frame.empty:
        report_lines.append("- No gating runs were evaluated.")
    else:
        gate_summary = gating_frame.agg(
            {
                "gate_mean": "mean",
                "corr_gate_uncertainty": "mean",
                "corr_gate_action_entropy": "mean",
                "corr_gate_reward": "mean",
            }
        )
        report_lines.append(
            f"- mean gate activation {gate_summary['gate_mean']:.6f}, mean corr(gate, uncertainty) {gate_summary['corr_gate_uncertainty']:.6f}, "
            f"mean corr(gate, action entropy) {gate_summary['corr_gate_action_entropy']:.6f}, mean corr(gate, reward) {gate_summary['corr_gate_reward']:.6f}"
        )
    report_lines.extend(
        [
            "",
            "## Claim Assessment",
            "",
            "- Bayesian uncertainty features are informative if shuffle and leave-one-feature-out materially degrade performance or alter Sharpe.",
            "- Standard PPO fails to exploit them robustly if the Bayes variants have unstable per-run rewards, wide intervals, or lose to simpler baselines/heuristics.",
            "- The learned attention variant should be discussed as an alternative mechanism, not as proof that the original gating solves the robustness issue.",
            "",
            "## Output Files",
            "",
            "- `outputs/diagnostics_neurips_revision/main_results_with_median_ci.csv`",
            "- `outputs/diagnostics_neurips_revision/shuffle_diagnostic.csv`",
            "- `outputs/diagnostics_neurips_revision/leave_one_bayesvol_feature_out.csv`",
            "- `outputs/diagnostics_neurips_revision/regime_analysis.csv`",
            "- `outputs/diagnostics_neurips_revision/gating_diagnostics.csv`",
            "- `outputs/diagnostics_neurips_revision/baseline_comparison.csv`",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
