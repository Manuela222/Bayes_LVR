from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bayesvol.cir import CIRParams
from src.bayesvol.ekf import CIRVolEKF
from src.utils.numeric import EPS, safe_sqrt


@dataclass
class BayesLVRFeatures:
    posterior_vol_mean: float
    posterior_vol_var: float
    expected_lvr_next: float
    var_lvr_next: float
    ci90_lvr_width: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.posterior_vol_mean,
                self.posterior_vol_var,
                self.expected_lvr_next,
                self.var_lvr_next,
                self.ci90_lvr_width,
            ],
            dtype=np.float32,
        )


class BayesVolFeatureEngine:
    def __init__(self, params: CIRParams, ci_z: float = 1.645):
        self.params = params
        self.ci_z = ci_z
        self.filter = CIRVolEKF(params)

    def reset(self) -> None:
        self.filter.reset()

    def update(self, observed_return: float, price: float, liquidity: float, portfolio_value: float) -> BayesLVRFeatures:
        state = self.filter.step(observed_return)
        coeff = self._lvr_coefficient(price=price, liquidity=liquidity, portfolio_value=portfolio_value)
        expected_lvr = coeff * state.mean
        var_lvr = (coeff ** 2) * state.var
        ci_width = 2.0 * self.ci_z * safe_sqrt(var_lvr)
        return BayesLVRFeatures(
            posterior_vol_mean=float(state.mean),
            posterior_vol_var=float(state.var),
            expected_lvr_next=float(expected_lvr),
            var_lvr_next=float(max(var_lvr, EPS)),
            ci90_lvr_width=float(ci_width),
        )

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "posterior_vol_mean",
            "posterior_vol_var",
            "expected_lvr_next",
            "var_lvr_next",
            "ci90_lvr_width",
        ]

    @staticmethod
    def _lvr_coefficient(price: float, liquidity: float, portfolio_value: float) -> float:
        safe_price = max(float(price), EPS)
        safe_portfolio = max(float(portfolio_value), EPS)
        safe_liquidity = max(float(liquidity), 0.0)
        return safe_liquidity * np.sqrt(safe_price) / (4.0 * safe_portfolio)
