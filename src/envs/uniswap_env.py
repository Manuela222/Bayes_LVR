from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.bayesvol.features import BayesVolFeatureEngine
from src.utils.numeric import EPS, ensure_finite_array, safe_log_return

try:
    import talib
    from talib import MA_Type
except Exception:  # pragma: no cover
    talib = None
    MA_Type = None


@dataclass
class EnvConfig:
    delta: float
    action_values: list[int]
    x: float
    gas_fee: float
    min_liquidity: float = 1e-9
    use_bayes_features: bool = False
    bayes_feature_normalizer: dict[str, dict[str, float]] | None = None


class UniswapV3BayesEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, market_data: pd.DataFrame, env_config: EnvConfig, bayes_engine: BayesVolFeatureEngine | None = None):
        super().__init__()
        self.df = market_data.copy().reset_index(drop=True)
        self.raw_price_series = self.df["price"].astype(float).to_numpy()
        self.raw_timestamps = self.df["timestamp"].tolist() if "timestamp" in self.df.columns else list(range(len(self.df)))
        self.env_config = env_config
        self.bayes_engine = bayes_engine
        self.delta = env_config.delta
        self.gas = env_config.gas_fee
        self.initial_x = float(env_config.x)
        self.bayes_feature_normalizer = env_config.bayes_feature_normalizer or {}
        self.tick_spacing = self._fee_to_tickspacing(self.delta)
        self.action_values = np.asarray(env_config.action_values, dtype=int)
        self.action_space = spaces.Discrete(len(self.action_values))
        self.width = int(self.action_values[1] if len(self.action_values) > 1 else self.action_values[0])
        self._build_indicators_like_finai()

        self.feature_names = [
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
        ]
        if env_config.use_bayes_features:
            self.feature_names.extend(BayesVolFeatureEngine.feature_names())
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_names),), dtype=np.float32)
        self.reset()

    def _build_indicators_like_finai(self) -> None:
        prices = self.raw_price_series
        self.open_price, self.high_price, self.low_price, self.close_price = self._calculate_candles(prices, 12)
        log_returns = np.log(np.maximum(prices[1:], EPS) / np.maximum(prices[:-1], EPS))
        self.log_returns = np.concatenate([[0.0], log_returns])
        ew_sigma = pd.DataFrame({"return": log_returns}).ewm(alpha=0.05).std()["return"].to_numpy()
        ew_sigma_full = np.concatenate([[0.0], np.nan_to_num(ew_sigma, nan=0.0)])

        ma_window_0 = 24
        ma_window_max = 168
        moving_average_24 = pd.DataFrame({"price": prices}).rolling(ma_window_0).mean().to_numpy()
        moving_average_168 = pd.DataFrame({"price": prices}).rolling(ma_window_max).mean().to_numpy()

        if talib is not None:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices, matype=MA_Type.T3)
            adxr = talib.ADX(self.high_price, self.low_price, self.close_price, timeperiod=14)
            bop = talib.BOP(self.open_price, self.high_price, self.low_price, self.close_price)
            dx = talib.DX(self.high_price, self.low_price, self.close_price, timeperiod=14)
        else:
            rolling_std = pd.Series(prices).rolling(ma_window_0).std().to_numpy()
            bb_middle = moving_average_24[:, 0]
            bb_upper = bb_middle + 2.0 * np.nan_to_num(rolling_std, nan=0.0)
            bb_lower = bb_middle - 2.0 * np.nan_to_num(rolling_std, nan=0.0)
            adxr = np.zeros_like(prices)
            bop = np.zeros_like(prices)
            dx = np.zeros_like(prices)

        # Match the FinAI starter-kit preprocessing when enough data is available.
        cut = min(ma_window_max, max(len(prices) - 2, 0))
        self.timestamps = self.raw_timestamps[cut:]
        self.price_series = prices[cut:]
        self.log_returns_series = self.log_returns[cut:]
        self.ew_sigma = ew_sigma_full[cut:]
        self.ma24 = np.nan_to_num(moving_average_24[cut:, 0], nan=0.0)
        self.ma168 = np.nan_to_num(moving_average_168[cut:, 0], nan=0.0)
        self.bb_upper = np.nan_to_num(bb_upper[cut:], nan=0.0)
        self.bb_middle = np.nan_to_num(bb_middle[cut:], nan=0.0)
        self.bb_lower = np.nan_to_num(bb_lower[cut:], nan=0.0)
        self.adxr = np.nan_to_num(adxr[cut:], nan=0.0)
        self.bop = np.nan_to_num(bop[cut:], nan=0.0)
        self.dx = np.nan_to_num(dx[cut:], nan=0.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.count = 0
        self.x = self.initial_x
        self.cumul_reward = 0.0
        self.cumul_fee = 0.0
        self.history: list[dict] = []
        pt = float(self.price_series[self.count])
        tick = self._price2tick(pt)
        self.width = int(self.action_values[1] if len(self.action_values) > 1 else self.action_values[0])
        tl, tu = tick - self.tick_spacing * self.width, tick + self.tick_spacing * self.width
        self.pl = self._tick2price(tl)
        self.pu = self._tick2price(tu)
        self.liquidity = self.x / max((1.0 / np.sqrt(pt)) - (1.0 / np.sqrt(self.pu)), EPS)
        self.y = self.liquidity * (np.sqrt(pt) - np.sqrt(self.pl))
        if self.bayes_engine is not None:
            self.bayes_engine.reset()
        observation = self._get_observation(self._default_bayes_features())
        return observation, {}

    def step(self, action_index: int):
        action = int(self.action_values[int(action_index)])
        self.count += 1

        prev_x = self.x
        prev_y = self.y
        pt_prev = float(self.price_series[self.count - 1])
        pt = float(self.price_series[self.count])
        tick = self._price2tick(pt)
        pl_prev, pu_prev = self.pl, self.pu
        xt, yt = self._calculate_xy(pt, pl_prev, pu_prev)
        forced_rebalance = xt == 0.0 or yt == 0.0

        if forced_rebalance:
            if action == 0:
                pl = self.pl
                pu = self.pu
                if xt == 0.0 and yt != 0.0:
                    self.liquidity = yt / max(np.sqrt(pu) - np.sqrt(pl), EPS)
                elif yt == 0.0 and xt != 0.0:
                    self.liquidity = xt / max((1.0 / np.sqrt(pl)) - (1.0 / np.sqrt(pu)), EPS)
                else:
                    self.liquidity = 0.0
            else:
                self.width = action
                tl, tu = tick - self.tick_spacing * self.width, tick + self.tick_spacing * self.width
                pl, pu = self._tick2price(tl), self._tick2price(tu)
                if yt == 0.0:
                    xt = xt / 2.0
                    yt = xt * pt
                elif xt == 0.0:
                    yt = yt / 2.0
                    xt = yt / max(pt, EPS)
                self.liquidity = xt / max((1.0 / np.sqrt(pt)) - (1.0 / np.sqrt(pu)), EPS)
        else:
            if action != 0:
                self.width = action
                tl, tu = tick - self.tick_spacing * self.width, tick + self.tick_spacing * self.width
                pl, pu = self._tick2price(tl), self._tick2price(tu)
                self.liquidity = xt / max((1.0 / np.sqrt(pt)) - (1.0 / np.sqrt(pu)), EPS)
            else:
                pl = pl_prev
                pu = pu_prev

        self.x = xt
        self.y = yt
        self.pl = pl
        self.pu = pu

        gas_fee = self._indicator(action) * self.gas
        fees = self._compute_fees(pt_prev, pt, pl, pu)
        sigma = float(self.ew_sigma[self.count])
        ground_truth_lvr = self._compute_ground_truth_lvr(price=pt, sigma=sigma)
        portfolio_value = float(self.x * pt + self.y)
        bayes_features = self._compute_bayes_features(pt, portfolio_value, sigma)
        reward = float(fees - gas_fee - ground_truth_lvr)
        self.cumul_reward += reward
        self.cumul_fee += fees

        terminated = self.count >= len(self.price_series) - 2
        truncated = self.liquidity <= self.env_config.min_liquidity
        observation = self._get_observation(bayes_features)
        info = {
            "timestamp": str(self.timestamps[self.count]),
            "price": pt,
            "prev_price": pt_prev,
            "fee": fees,
            "gas_fee": gas_fee,
            "lvr": ground_truth_lvr,
            "lvr_ground_truth": ground_truth_lvr,
            "width": self.width,
            "forced_rebalance": forced_rebalance,
            "bayes": bayes_features,
            "portfolio_value": portfolio_value,
            "sigma_empirical": sigma,
        }
        self.history.append(
            {
                "step": self.count,
                "timestamp": info["timestamp"],
                "reward": reward,
                "fee": fees,
                "gas_fee": gas_fee,
                "lvr": ground_truth_lvr,
                "lvr_ground_truth": ground_truth_lvr,
                "action": action,
                "width": self.width,
                "forced_rebalance": forced_rebalance,
                "portfolio_value": portfolio_value,
                "price": pt,
                "prev_price": pt_prev,
                "prev_x": prev_x,
                "prev_y": prev_y,
                "sigma_empirical": sigma,
                "bayes": bayes_features,
                "observation": observation.tolist(),
            }
        )
        return observation, reward, terminated, truncated, info

    def _compute_bayes_features(self, price: float, portfolio_value: float, empirical_sigma: float) -> dict[str, float]:
        bayes = self._default_bayes_features()
        bayes["empirical_vol_proxy"] = float(empirical_sigma)
        if self.bayes_engine is None:
            return bayes
        observed_return = float(self.log_returns_series[self.count])
        features = self.bayes_engine.update(
            observed_return=observed_return,
            price=price,
            liquidity=self.liquidity,
            portfolio_value=portfolio_value,
        )
        bayes.update(dict(zip(BayesVolFeatureEngine.feature_names(), features.as_array().tolist())))
        return bayes

    @staticmethod
    def _default_bayes_features() -> dict[str, float]:
        return {
            "posterior_vol_mean": 0.0,
            "posterior_vol_var": 0.0,
            "expected_lvr_next": 0.0,
            "var_lvr_next": 0.0,
            "ci90_lvr_width": 0.0,
            "empirical_vol_proxy": 0.0,
        }

    def _get_observation(self, bayes_features: dict[str, float]) -> np.ndarray:
        idx = self.count
        price = float(self.price_series[idx])
        tick = self._price2tick(price)
        values = [
            price,
            float(tick),
            float(self.width),
            float(self.liquidity),
            float(self.ew_sigma[idx]),
            float(self.ma24[idx]),
            float(self.ma168[idx]),
            float(self.bb_upper[idx]),
            float(self.bb_middle[idx]),
            float(self.bb_lower[idx]),
            float(self.adxr[idx]),
            float(self.bop[idx]),
            float(self.dx[idx]),
        ]
        if self.env_config.use_bayes_features:
            for name in BayesVolFeatureEngine.feature_names():
                values.append(float(self._normalize_bayes_feature(name, bayes_features[name])))
        return ensure_finite_array(values)

    def _normalize_bayes_feature(self, name: str, value: float) -> float:
        params = self.bayes_feature_normalizer.get(name)
        if not params:
            return float(value)
        scale = max(float(params.get("std", 1.0)), EPS)
        normalized = (float(value) - float(params.get("mean", 0.0))) / scale
        return float(np.clip(normalized, -10.0, 10.0))

    def _compute_ground_truth_lvr(self, price: float, sigma: float) -> float:
        vp = self.x * price + self.y
        ll = self.liquidity * sigma * sigma / 4.0 * np.sqrt(max(price, EPS))
        if vp != 0.0:
            return float(ll)
        return 1e9

    @staticmethod
    def _calculate_candles(prices: np.ndarray, interval_length: int):
        open_prices, high_prices, low_prices, close_prices = [], [], [], []
        lookback = interval_length - 1
        for i in range(len(prices)):
            start_idx = 0 if i < lookback else i - lookback
            data_slice = prices[start_idx : i + 1]
            open_prices.append(float(data_slice[0]))
            high_prices.append(float(np.max(data_slice)))
            low_prices.append(float(np.min(data_slice)))
            close_prices.append(float(data_slice[-1]))
        return tuple(np.asarray(series, dtype=np.float64) for series in [open_prices, high_prices, low_prices, close_prices])

    @staticmethod
    def _price2tick(price: float) -> int:
        return math.floor(math.log(max(price, EPS), 1.0001))

    @staticmethod
    def _tick2price(tick: int) -> float:
        return 1.0001**tick

    @staticmethod
    def _fee_to_tickspacing(fee_tier: float) -> int:
        if fee_tier == 0.05:
            return 10
        if fee_tier == 0.30:
            return 60
        if fee_tier == 1.00:
            return 200
        raise ValueError(f"Unsupported fee tier: {fee_tier}")

    def _compute_fees(self, prev_price: float, price: float, pl: float, pu: float) -> float:
        if prev_price <= price:
            if prev_price >= pu or price <= pl:
                return 0.0
            p_prime = min(price, pu)
            p = max(prev_price, pl)
        else:
            if prev_price <= pl or price >= pu:
                return 0.0
            p_prime = max(price, pl)
            p = min(prev_price, pu)
        return self._calculate_fee(p, p_prime)

    def _calculate_fee(self, p: float, p_prime: float) -> float:
        if p <= p_prime:
            return float((self.delta / (1.0 - self.delta)) * self.liquidity * (math.sqrt(p_prime) - math.sqrt(p)))
        return float((self.delta / (1.0 - self.delta)) * self.liquidity * ((1.0 / math.sqrt(p_prime)) - (1.0 / math.sqrt(p))) * p_prime)

    def _indicator(self, action: int) -> int:
        return 1 if action != 0 else 0

    def _calculate_xy(self, price: float, pl: float, pu: float) -> tuple[float, float]:
        if price <= pl:
            x = self.liquidity * (1.0 / math.sqrt(pl) - 1.0 / math.sqrt(pu))
            y = 0.0
        elif price >= pu:
            x = 0.0
            y = self.liquidity * (math.sqrt(pu) - math.sqrt(pl))
        else:
            x = self.liquidity * (1.0 / math.sqrt(price) - 1.0 / math.sqrt(pu))
            y = self.liquidity * (math.sqrt(price) - math.sqrt(pl))
        return float(x), float(y)
