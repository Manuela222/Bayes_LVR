from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bayesvol.cir import CIRParams
from src.utils.numeric import EPS


@dataclass
class BayesianVolState:
    mean: float
    var: float


class CIRVolEKF:
    def __init__(self, params: CIRParams, initial_mean: float | None = None, initial_var: float | None = None):
        self.params = params
        self.initial_mean = float(initial_mean if initial_mean is not None else max(params.theta, params.eps))
        self.initial_var = float(initial_var if initial_var is not None else max(self.initial_mean * 0.25, params.eps))
        self.reset()

    def reset(self) -> BayesianVolState:
        self.state = BayesianVolState(mean=self.initial_mean, var=self.initial_var)
        return self.state

    def step(self, observed_return: float) -> BayesianVolState:
        p = self.params
        x_prev = max(self.state.mean, p.eps)
        var_prev = max(self.state.var, p.eps)

        predicted_mean = x_prev + p.kappa * (p.theta - x_prev) * p.dt
        predicted_mean = float(max(predicted_mean, p.eps))

        jacobian = max(1.0 - p.kappa * p.dt, 0.0)
        process_var = (p.xi ** 2) * max(x_prev, p.eps) * p.dt
        predicted_var = float(max((jacobian ** 2) * var_prev + process_var, p.eps))

        measurement = max(float(observed_return) ** 2, p.eps)
        measurement_jacobian = 1.0
        measurement_var = float(max(2.0 * predicted_mean ** 2, p.eps))
        innovation_var = float(max(measurement_jacobian ** 2 * predicted_var + measurement_var, p.eps))
        kalman_gain = predicted_var * measurement_jacobian / innovation_var

        updated_mean = predicted_mean + kalman_gain * (measurement - predicted_mean)
        updated_mean = float(max(updated_mean, p.eps))
        updated_var = float(max((1.0 - kalman_gain * measurement_jacobian) * predicted_var, p.eps))

        if not np.isfinite(updated_mean) or not np.isfinite(updated_var):
            updated_mean = predicted_mean
            updated_var = predicted_var

        self.state = BayesianVolState(mean=updated_mean, var=updated_var)
        return self.state
