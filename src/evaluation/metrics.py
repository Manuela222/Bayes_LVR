from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.utils.numeric import EPS, safe_corr, safe_rank_corr


def compute_rollout_metrics(history: list[dict]) -> dict[str, float | dict[str, int]]:
    if not history:
        return {}
    rewards = np.asarray([item["reward"] for item in history], dtype=float)
    fees = np.asarray([item["fee"] for item in history], dtype=float)
    lvr = np.asarray([item["lvr_ground_truth"] for item in history], dtype=float)
    gas = np.asarray([item["gas_fee"] for item in history], dtype=float)
    widths = np.asarray([item["width"] for item in history], dtype=float)
    uncertainty = np.asarray([item["bayes"]["posterior_vol_var"] for item in history], dtype=float)
    posterior_mean = np.asarray([item["bayes"]["posterior_vol_mean"] for item in history], dtype=float)
    empirical_sigma = np.asarray([item.get("sigma_empirical", item["bayes"].get("empirical_vol_proxy", 0.0)) for item in history], dtype=float)
    action_counts = Counter(str(item["action"]) for item in history)

    sharpe = 0.0
    if rewards.size > 1 and float(np.std(rewards)) > EPS:
        sharpe = float(np.mean(rewards) / np.std(rewards) * np.sqrt(24.0 * 365.0))

    total_lvr = float(np.sum(lvr))
    total_fees = float(np.sum(fees))
    robust_corr = 0.5 * safe_corr(uncertainty, widths) + 0.5 * safe_rank_corr(uncertainty, widths)
    return {
        "cumulative_reward": float(np.sum(rewards)),
        "sharpe_ratio": sharpe,
        "total_fees": total_fees,
        "total_lvr": total_lvr,
        "gas_cost": float(np.sum(gas)),
        "fees_to_lvr_ratio": float(total_fees / max(total_lvr, EPS)),
        "action_distribution": dict(action_counts),
        "corr_lvr_uncertainty_width": float(robust_corr),
        "bayes_feature_stats": compute_bayes_feature_diagnostics(history),
        "policy_diagnostics": compute_policy_diagnostics(history),
    }


def compute_bayes_feature_diagnostics(history: list[dict]) -> dict:
    posterior_mean = np.asarray([item["bayes"]["posterior_vol_mean"] for item in history], dtype=float)
    posterior_var = np.asarray([item["bayes"]["posterior_vol_var"] for item in history], dtype=float)
    empirical_sigma = np.asarray([item.get("sigma_empirical", item["bayes"].get("empirical_vol_proxy", 0.0)) for item in history], dtype=float)
    widths = np.asarray([item["width"] for item in history], dtype=float)
    actions = np.asarray([item["action"] for item in history], dtype=float)
    if len(posterior_var) == 0:
        return {}

    frame = pd.DataFrame({"posterior_var": posterior_var, "width": widths, "action": actions})
    try:
        frame["quantile"] = pd.qcut(frame["posterior_var"], q=min(4, max(1, frame["posterior_var"].nunique())), duplicates="drop")
        by_quantile = (
            frame.groupby("quantile", observed=False)["width"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
            .astype({"count": int})
        )
        action_by_quantile = {
            str(row["quantile"]): {
                "count": int(row["count"]),
                "width_mean": float(row["mean"]),
                "width_min": float(row["min"]),
                "width_max": float(row["max"]),
            }
            for _, row in by_quantile.iterrows()
        }
        action_counts = {}
        for quantile, group in frame.groupby("quantile", observed=False):
            action_counts[str(quantile)] = {str(int(action)): int(count) for action, count in group["action"].value_counts().sort_index().items()}
    except Exception:
        action_by_quantile = {}
        action_counts = {}

    return {
        "posterior_vol_mean_summary": _summary_stats(posterior_mean),
        "posterior_vol_var_summary": _summary_stats(posterior_var),
        "corr_posterior_mean_empirical_sigma": float(safe_corr(posterior_mean, empirical_sigma)),
        "corr_posterior_var_empirical_sigma": float(safe_corr(posterior_var, empirical_sigma)),
        "corr_feature_reward": {
            "posterior_vol_mean": float(safe_corr(posterior_mean, np.asarray([item["reward"] for item in history], dtype=float))),
            "posterior_vol_var": float(safe_corr(posterior_var, np.asarray([item["reward"] for item in history], dtype=float))),
            "expected_lvr_next": float(safe_corr(np.asarray([item["bayes"]["expected_lvr_next"] for item in history], dtype=float), np.asarray([item["reward"] for item in history], dtype=float))),
        },
        "corr_feature_action": {
            "posterior_vol_mean": float(safe_corr(posterior_mean, actions)),
            "posterior_vol_var": float(safe_corr(posterior_var, actions)),
            "expected_lvr_next": float(safe_corr(np.asarray([item["bayes"]["expected_lvr_next"] for item in history], dtype=float), actions)),
        },
        "action_distribution_by_posterior_var_quantile": action_by_quantile,
        "action_counts_by_posterior_var_quantile": action_counts,
    }


def summarize_gate(gates: np.ndarray) -> dict[str, float]:
    return _summary_stats(gates.astype(float))


def compute_policy_diagnostics(history: list[dict]) -> dict:
    if not history:
        return {}
    actions = np.asarray([item["action"] for item in history], dtype=float)
    widths = np.asarray([item["width"] for item in history], dtype=float)
    return {
        "unique_actions": int(np.unique(actions).size),
        "action_std": float(np.std(actions)),
        "width_std": float(np.std(widths)),
    }


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }
