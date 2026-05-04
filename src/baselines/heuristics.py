from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.metrics import compute_rollout_metrics
from src.volatility.features import FeatureEngineSpec, build_feature_engine


@dataclass
class HeuristicResult:
    history: list[dict]
    metrics: dict


def run_heuristic_rollout(env, policy_name: str) -> HeuristicResult:
    obs, _ = env.reset()
    terminated = False
    truncated = False
    feature_names = list(env.unwrapped.feature_names)
    action_values = np.asarray(env.unwrapped.action_values, dtype=int)
    positive_widths = np.asarray([value for value in action_values if value > 0], dtype=int)
    if positive_widths.size == 0:
        positive_widths = action_values

    while not (terminated or truncated):
        action_idx = _select_action_index(
            obs=np.asarray(obs, dtype=float),
            feature_names=feature_names,
            current_width=int(env.unwrapped.width),
            action_values=action_values,
            positive_widths=positive_widths,
            policy_name=policy_name,
        )
        obs, _, terminated, truncated, _ = env.step(action_idx)
    history = env.unwrapped.history
    return HeuristicResult(history=history, metrics=compute_rollout_metrics(history))


def _select_action_index(
    obs: np.ndarray,
    feature_names: list[str],
    current_width: int,
    action_values: np.ndarray,
    positive_widths: np.ndarray,
    policy_name: str,
) -> int:
    feature_map = {name: float(obs[idx]) for idx, name in enumerate(feature_names)}
    sigma2 = max(feature_map.get("posterior_vol_mean", 0.0), 1e-12)
    uncertainty = max(feature_map.get("posterior_vol_var", 0.0), 1e-12)
    lvr_proxy = max(feature_map.get("expected_lvr_next", 0.0), 0.0)

    if policy_name == "garch_heuristic":
        risk_score = sigma2 + 0.5 * np.sqrt(uncertainty)
        normalized = np.clip(risk_score / max(np.quantile([sigma2, sigma2 + uncertainty, risk_score], 0.75), 1e-8), 0.0, 2.0)
        target = positive_widths[min(len(positive_widths) - 1, int(round(normalized * (len(positive_widths) - 1) / 2.0)))]
    elif policy_name == "cartea_style_control":
        control_signal = lvr_proxy + 0.25 * np.sqrt(uncertainty)
        if control_signal <= 1e-6:
            target = positive_widths[0]
        else:
            rank = int(np.clip(np.floor(np.log10(1.0 + control_signal * 1e4)), 0, len(positive_widths) - 1))
            target = positive_widths[rank]
    else:
        raise ValueError(f"Unsupported heuristic policy: {policy_name}")

    if current_width == int(target) and 0 in action_values:
        return int(np.where(action_values == 0)[0][0])
    return int(np.where(action_values == target)[0][0])
