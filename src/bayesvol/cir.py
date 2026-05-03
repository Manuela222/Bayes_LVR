from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.utils.numeric import EPS


@dataclass
class CIRParams:
    kappa: float = 0.12
    theta: float = 1e-4
    xi: float = 0.08
    dt: float = 1.0
    eps: float = EPS

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def estimate_cir_params(returns: np.ndarray, dt: float = 1.0, eps: float = EPS) -> CIRParams:
    squared = np.maximum(np.asarray(returns, dtype=float) ** 2, eps)
    if len(squared) < 3:
        return CIRParams(dt=dt, eps=eps)

    x_prev = squared[:-1]
    x_next = squared[1:]
    beta, alpha = np.polyfit(x_prev, x_next, deg=1)
    beta = float(np.clip(beta, 0.0, 0.999))
    alpha = float(max(alpha, eps))

    kappa = float(max((1.0 - beta) / max(dt, eps), 1e-4))
    theta = float(max(alpha / max(kappa * dt, eps), eps))
    residuals = x_next - (alpha + beta * x_prev)
    denom = np.sqrt(np.maximum(x_prev, eps))
    scaled = residuals / denom
    xi = float(max(np.std(scaled) / np.sqrt(max(dt, eps)), 1e-4))

    return CIRParams(kappa=kappa, theta=theta, xi=xi, dt=dt, eps=eps)
