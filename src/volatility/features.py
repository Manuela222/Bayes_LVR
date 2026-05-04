from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.bayesvol.cir import CIRParams, estimate_cir_params
from src.bayesvol.features import BayesVolFeatureEngine
from src.utils.numeric import EPS, safe_log_return, safe_sqrt


FEATURE_NAMES = [
    "posterior_vol_mean",
    "posterior_vol_var",
    "expected_lvr_next",
    "var_lvr_next",
    "ci90_lvr_width",
]


@dataclass
class FeatureEngineSpec:
    source: str
    params: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"source": self.source, "params": dict(self.params)}


@dataclass
class VolatilityFeatureVector:
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


class BaseFeatureEngine:
    source_name = "none"

    def reset(self) -> None:
        return None

    def update(self, observed_return: float, price: float, liquidity: float, portfolio_value: float) -> VolatilityFeatureVector:
        raise NotImplementedError

    @staticmethod
    def feature_names() -> list[str]:
        return list(FEATURE_NAMES)

    @staticmethod
    def _lvr_coefficient(price: float, liquidity: float, portfolio_value: float) -> float:
        safe_price = max(float(price), EPS)
        safe_portfolio = max(float(portfolio_value), EPS)
        safe_liquidity = max(float(liquidity), 0.0)
        return safe_liquidity * np.sqrt(safe_price) / (4.0 * safe_portfolio)

    def _vector_from_variance(self, sigma2: float, variance_uncertainty: float, price: float, liquidity: float, portfolio_value: float) -> VolatilityFeatureVector:
        coeff = self._lvr_coefficient(price=price, liquidity=liquidity, portfolio_value=portfolio_value)
        expected_lvr = coeff * sigma2
        var_lvr = max((coeff ** 2) * variance_uncertainty, EPS)
        ci_width = 2.0 * 1.645 * safe_sqrt(var_lvr)
        return VolatilityFeatureVector(
            posterior_vol_mean=float(max(sigma2, EPS)),
            posterior_vol_var=float(max(variance_uncertainty, EPS)),
            expected_lvr_next=float(expected_lvr),
            var_lvr_next=float(var_lvr),
            ci90_lvr_width=float(ci_width),
        )


class EWMAVolFeatureEngine(BaseFeatureEngine):
    source_name = "ewma"

    def __init__(self, alpha: float = 0.05, initial_variance: float = 1e-4):
        self.alpha = float(alpha)
        self.initial_variance = float(max(initial_variance, EPS))
        self.sigma2 = self.initial_variance

    def reset(self) -> None:
        self.sigma2 = self.initial_variance

    def update(self, observed_return: float, price: float, liquidity: float, portfolio_value: float) -> VolatilityFeatureVector:
        prev = self.sigma2
        self.sigma2 = self.alpha * float(observed_return) ** 2 + (1.0 - self.alpha) * self.sigma2
        uncertainty = max((self.alpha ** 2) * (prev ** 2 + self.sigma2 ** 2), EPS)
        return self._vector_from_variance(self.sigma2, uncertainty, price, liquidity, portfolio_value)


class GARCHVolFeatureEngine(BaseFeatureEngine):
    source_name = "garch"

    def __init__(self, omega: float, alpha: float, beta: float, initial_variance: float):
        self.omega = float(max(omega, EPS))
        self.alpha = float(np.clip(alpha, 1e-4, 0.3))
        self.beta = float(np.clip(beta, 0.5, 0.999))
        if self.alpha + self.beta >= 0.999:
            self.beta = 0.999 - self.alpha
        self.initial_variance = float(max(initial_variance, EPS))
        self.sigma2 = self.initial_variance

    def reset(self) -> None:
        self.sigma2 = self.initial_variance

    def update(self, observed_return: float, price: float, liquidity: float, portfolio_value: float) -> VolatilityFeatureVector:
        shock = float(observed_return) ** 2
        self.sigma2 = self.omega + self.alpha * shock + self.beta * self.sigma2
        persistence = max(self.alpha + self.beta, EPS)
        uncertainty = max(2.0 * (self.sigma2 ** 2) * persistence ** 2, EPS)
        return self._vector_from_variance(self.sigma2, uncertainty, price, liquidity, portfolio_value)


class NullFeatureEngine(BaseFeatureEngine):
    source_name = "none"

    def update(self, observed_return: float, price: float, liquidity: float, portfolio_value: float) -> VolatilityFeatureVector:
        return self._vector_from_variance(EPS, EPS, price, liquidity, portfolio_value)


def get_feature_source(config: dict) -> str:
    feature_cfg = config.get("features", {})
    if "feature_source" in feature_cfg:
        return str(feature_cfg["feature_source"]).lower()
    if feature_cfg.get("use_bayes_features", False):
        return "bayes"
    return "none"


def uses_volatility_features(config: dict) -> bool:
    return get_feature_source(config) != "none"


def fit_feature_engine_spec(train_frame, config: dict) -> FeatureEngineSpec | None:
    source = get_feature_source(config)
    if source == "none":
        return None

    prices = train_frame["price"].astype(float).to_numpy()
    returns = np.array([safe_log_return(prices[i], prices[i - 1]) for i in range(1, len(prices))], dtype=float)
    variance = float(max(np.var(returns) if returns.size else 1e-4, EPS))

    if source == "bayes":
        return FeatureEngineSpec(source="bayes", params=estimate_cir_params(returns, dt=1.0).to_dict())
    if source == "ewma":
        alpha = float(config.get("features", {}).get("ewma_alpha", 0.05))
        return FeatureEngineSpec(source="ewma", params={"alpha": alpha, "initial_variance": variance})
    if source == "garch":
        alpha = float(config.get("features", {}).get("garch_alpha", 0.08))
        beta = float(config.get("features", {}).get("garch_beta", 0.9))
        squared = returns ** 2
        if squared.size > 2 and float(np.std(squared[:-1])) > EPS and float(np.std(squared[1:])) > EPS:
            persistence_hint = float(np.corrcoef(squared[:-1], squared[1:])[0, 1])
            if np.isfinite(persistence_hint):
                beta = float(np.clip(0.75 + 0.2 * persistence_hint, 0.75, 0.97))
                alpha = float(np.clip(0.18 - 0.08 * persistence_hint, 0.03, 0.15))
        omega = max(variance * max(1.0 - alpha - beta, 1e-3), EPS)
        return FeatureEngineSpec(
            source="garch",
            params={"omega": omega, "alpha": alpha, "beta": beta, "initial_variance": variance},
        )
    raise ValueError(f"Unsupported feature source: {source}")


def build_feature_engine(spec: FeatureEngineSpec | None) -> BaseFeatureEngine | None:
    if spec is None:
        return None
    if spec.source == "bayes":
        return BayesVolFeatureEngine(CIRParams(**spec.params))
    if spec.source == "ewma":
        return EWMAVolFeatureEngine(**spec.params)
    if spec.source == "garch":
        return GARCHVolFeatureEngine(**spec.params)
    if spec.source == "none":
        return NullFeatureEngine()
    raise ValueError(f"Unsupported feature source: {spec.source}")


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)


def feature_indices(all_feature_names: list[str]) -> list[int]:
    return [idx for idx, name in enumerate(all_feature_names) if name in FEATURE_NAMES]
