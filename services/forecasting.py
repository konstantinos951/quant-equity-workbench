from __future__ import annotations

from pathlib import Path
import os
from typing import Any
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tools.sm_exceptions import ConvergenceWarning, EstimationWarning

_MPL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

from arch import arch_model

from services.models import ForecastConfig, StockDataset


class ForecastModelError(RuntimeError):
    pass


def _fit_autoregressive_mean_model(returns: pd.Series) -> dict[str, float]:
    try:
        fitted = AutoReg(returns, lags=1, trend="c", old_names=False).fit()
        intercept = float(fitted.params.iloc[0])
        phi = float(fitted.params.iloc[1]) if len(fitted.params) > 1 else 0.0
        phi = float(np.clip(phi, -0.35, 0.35))
        recent_mean = float(np.clip(returns.ewm(span=20, adjust=False).mean().iloc[-1], -0.01, 0.01))
        long_mean = float(np.clip(returns.mean(), -0.01, 0.01))
        return {
            "intercept": intercept,
            "phi": phi,
            "recent_mean": recent_mean,
            "long_mean": long_mean,
        }
    except Exception:
        mean_value = float(returns.mean())
        return {
            "intercept": mean_value,
            "phi": 0.0,
            "recent_mean": mean_value,
            "long_mean": mean_value,
        }


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total_weight = float(weights.sum())
    if total_weight <= 0:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        return mean, max(std, 1e-6)

    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, max(np.sqrt(variance), 1e-6)


def _fit_markov_model(returns: pd.Series, desired_regimes: int) -> dict[str, Any]:
    last_error: Exception | None = None

    for regimes in (desired_regimes, 2):
        try:
            model = MarkovRegression(
                returns,
                k_regimes=regimes,
                trend="c",
                switching_variance=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EstimationWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                result = model.fit(search_reps=20, em_iter=10, disp=False)

            smoothed = result.smoothed_marginal_probabilities.copy()
            filtered = result.filtered_marginal_probabilities.copy()
            transition = np.asarray(model.regime_transition_matrix(result.params))[:, :, 0]
            transition = np.clip(transition, 1e-10, None)
            transition = transition / transition.sum(axis=0, keepdims=True)

            regime_stats = []
            returns_array = returns.to_numpy(dtype=float)
            probs_array = smoothed.to_numpy(dtype=float)
            for regime in range(regimes):
                mean_return, std_return = _weighted_stats(returns_array, probs_array[:, regime])
                regime_stats.append(
                    {
                        "regime": regime,
                        "mean_return": mean_return,
                        "std_return": std_return,
                    }
                )

            order = np.argsort([item["mean_return"] for item in regime_stats])
            sorted_transition = transition[np.ix_(order, order)]
            sorted_smoothed = smoothed.iloc[:, order].copy()
            sorted_filtered = filtered.iloc[:, order].copy()

            labels = ["Bear", "Neutral", "Bull"] if regimes == 3 else ["Defensive", "Risk-On"]
            sorted_stats = [regime_stats[index] for index in order]
            if len(labels) != len(sorted_stats):
                labels = [f"Regime {index + 1}" for index in range(len(sorted_stats))]

            sorted_smoothed.columns = labels
            sorted_filtered.columns = labels
            for label, stats in zip(labels, sorted_stats):
                stats["label"] = label

            current_probs = sorted_filtered.iloc[-1].to_numpy(dtype=float)
            current_probs = current_probs / current_probs.sum()

            return {
                "model_name": "MarkovRegression",
                "labels": labels,
                "stats": sorted_stats,
                "smoothed_probabilities": sorted_smoothed,
                "filtered_probabilities": sorted_filtered,
                "transition_matrix": sorted_transition,
                "current_probabilities": current_probs,
            }
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise ForecastModelError("Το Markov regime model δεν κατάφερε να συγκλίνει.") from last_error


def _fallback_empirical_markov(returns: pd.Series, regimes: int = 3) -> dict[str, Any]:
    signal = (
        returns.ewm(span=20, adjust=False).mean()
        / returns.ewm(span=20, adjust=False).std().replace(0, np.nan)
    ).fillna(0.0)

    ranked = signal.rank(method="first")
    bins = pd.qcut(ranked, q=regimes, labels=False, duplicates="drop")
    bins = bins.astype(int)

    labels = ["Bear", "Neutral", "Bull"][: int(bins.max()) + 1]
    regime_matrix = pd.get_dummies(bins).reindex(columns=range(len(labels)), fill_value=0).astype(float)
    regime_matrix.index = returns.index
    regime_matrix.columns = labels

    transition_counts = np.ones((len(labels), len(labels)))
    states = bins.to_numpy(dtype=int)
    for prev_state, next_state in zip(states[:-1], states[1:]):
        transition_counts[next_state, prev_state] += 1
    transition_matrix = transition_counts / transition_counts.sum(axis=0, keepdims=True)

    stats = []
    returns_array = returns.to_numpy(dtype=float)
    probs_array = regime_matrix.to_numpy(dtype=float)
    for column, label in enumerate(labels):
        mean_return, std_return = _weighted_stats(returns_array, probs_array[:, column])
        stats.append({"regime": column, "label": label, "mean_return": mean_return, "std_return": std_return})

    current_probabilities = regime_matrix.iloc[-1].to_numpy(dtype=float)
    current_probabilities = current_probabilities / current_probabilities.sum()

    return {
        "model_name": "Empirical Markov fallback",
        "labels": labels,
        "stats": stats,
        "smoothed_probabilities": regime_matrix,
        "filtered_probabilities": regime_matrix,
        "transition_matrix": transition_matrix,
        "current_probabilities": current_probabilities,
    }


def _fit_volatility_candidate(
    returns: pd.Series,
    horizon: int,
    label: str,
    *,
    asymmetry_order: int = 0,
) -> dict[str, Any]:
    scaled_returns = returns * 100
    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=asymmetry_order,
        q=1,
        dist="t",
        rescale=False,
    )
    result = model.fit(disp="off", show_warning=False)
    forecast_method = "analytic" if asymmetry_order == 0 else "simulation"
    forecast_kwargs: dict[str, Any] = {"horizon": horizon, "method": forecast_method}
    if forecast_method == "simulation":
        forecast_kwargs["simulations"] = 400
    forecasts = result.forecast(**forecast_kwargs)
    variance_path = forecasts.variance.iloc[-1].to_numpy(dtype=float) / 10_000.0
    sigma_path = np.sqrt(np.maximum(variance_path, 1e-10))

    nu = float(result.params.get("nu", 8.0))
    if not np.isfinite(nu) or nu <= 2:
        nu = 8.0

    conditional_sigma = pd.Series(
        np.maximum(result.conditional_volatility.to_numpy(dtype=float) / 100.0, 1e-8),
        index=returns.index,
        name=label,
    )
    std_resid = (
        (result.resid / result.conditional_volatility)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    return {
        "label": label,
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
        "sigma_path": sigma_path,
        "degrees_of_freedom": nu,
        "params": result.params.to_dict(),
        "std_resid": std_resid,
        "conditional_sigma": conditional_sigma,
    }


def _ewma_volatility_fallback(returns: pd.Series, horizon: int) -> dict[str, Any]:
    lambda_ = 0.94
    sigma_sq = np.zeros(len(returns), dtype=float)
    sigma_sq[0] = max(float(returns.var(ddof=0)), 1e-8)
    values = returns.to_numpy(dtype=float)
    for index in range(1, len(values)):
        sigma_sq[index] = lambda_ * sigma_sq[index - 1] + (1 - lambda_) * values[index - 1] ** 2
    sigma_series = pd.Series(np.sqrt(np.maximum(sigma_sq, 1e-10)), index=returns.index, name="EWMA")
    std_resid = (returns / sigma_series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return {
        "sigma_path": np.full(horizon, float(sigma_series.iloc[-1])),
        "degrees_of_freedom": 8.0,
        "params": {"lambda": lambda_},
        "std_resid": std_resid if len(std_resid) else np.array([0.0]),
        "conditional_sigma": sigma_series,
        "models": [
            {
                "label": "EWMA fallback",
                "weight": 1.0,
                "aic": np.nan,
                "bic": np.nan,
            }
        ],
        "selected_model": "EWMA fallback",
    }


def _fit_volatility_ensemble(returns: pd.Series, horizon: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []

    for label, asymmetry_order in (
        ("GARCH(1,1)-t", 0),
        ("GJR-GARCH(1,1)-t", 1),
    ):
        try:
            candidates.append(
                _fit_volatility_candidate(
                    returns,
                    horizon=horizon,
                    label=label,
                    asymmetry_order=asymmetry_order,
                )
            )
        except Exception as exc:  # pragma: no cover
            errors.append(f"{label}: {exc}")

    if not candidates:
        return _ewma_volatility_fallback(returns, horizon)

    aics = np.array([candidate["aic"] for candidate in candidates], dtype=float)
    finite_mask = np.isfinite(aics)
    if not finite_mask.any():
        weights = np.full(len(candidates), 1.0 / len(candidates))
    else:
        min_aic = float(np.min(aics[finite_mask]))
        deltas = np.where(finite_mask, aics - min_aic, 10.0)
        weights = np.exp(-0.5 * deltas)
        weights = weights / weights.sum()

    sigma_path = np.zeros(horizon, dtype=float)
    conditional_sigma = pd.Series(0.0, index=returns.index, dtype=float)
    weighted_nu = 0.0
    pooled_residuals: list[np.ndarray] = []
    models_summary: list[dict[str, Any]] = []

    for weight, candidate in zip(weights, candidates):
        sigma_path += weight * np.asarray(candidate["sigma_path"], dtype=float)
        conditional_sigma = conditional_sigma.add(candidate["conditional_sigma"] * weight, fill_value=0.0)
        weighted_nu += weight * float(candidate["degrees_of_freedom"])
        if len(candidate["std_resid"]):
            pooled_residuals.append(np.asarray(candidate["std_resid"], dtype=float))
        models_summary.append(
            {
                "label": candidate["label"],
                "weight": float(weight),
                "aic": candidate["aic"],
                "bic": candidate["bic"],
            }
        )

    selected_model = models_summary[int(np.argmax(weights))]["label"]
    return {
        "sigma_path": np.maximum(sigma_path, 1e-8),
        "degrees_of_freedom": float(np.clip(weighted_nu, 3.0, 30.0)),
        "params": {candidate["label"]: candidate["params"] for candidate in candidates},
        "std_resid": np.concatenate(pooled_residuals) if pooled_residuals else np.array([0.0]),
        "conditional_sigma": conditional_sigma.replace(0, np.nan).bfill().ffill(),
        "models": models_summary,
        "selected_model": selected_model,
        "fit_errors": errors,
    }


def _estimate_jump_process(returns: pd.Series) -> dict[str, Any]:
    centered = returns - float(returns.mean())
    sigma = float(centered.std(ddof=0))
    if sigma <= 0:
        return {"jump_probability": 0.0, "jump_pool": np.array([0.0])}

    threshold = 2.5 * sigma
    tail = centered[np.abs(centered) >= threshold]
    if len(tail) < 8:
        ranked = centered.iloc[np.argsort(np.abs(centered.to_numpy()))[-12:]]
        tail = ranked

    jump_probability = float(np.clip(len(tail) / max(len(centered), 1), 0.005, 0.06))
    jump_pool = tail.to_numpy(dtype=float)
    if len(jump_pool) == 0:
        jump_pool = np.array([0.0])

    return {"jump_probability": jump_probability, "jump_pool": jump_pool}


def _sample_next_states(
    rng: np.random.Generator,
    transition_matrix: np.ndarray,
    current_states: np.ndarray,
) -> np.ndarray:
    draws = rng.random(len(current_states))
    cdf = np.cumsum(transition_matrix[:, current_states].T, axis=1)
    return (draws[:, None] > cdf).sum(axis=1)


def _zscore(series: pd.Series, lookback: int = 126) -> pd.Series:
    mean = series.rolling(lookback, min_periods=20).mean()
    std = series.rolling(lookback, min_periods=20).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _weighted_feature_score(latest_row: pd.Series, weights: dict[str, float]) -> float:
    contributions = []
    total_weight = 0.0
    for name, weight in weights.items():
        if name not in latest_row.index:
            continue
        value = latest_row[name]
        if value is None or not np.isfinite(value):
            continue
        contributions.append(weight * float(value))
        total_weight += abs(weight)
    if total_weight <= 0:
        return 0.0
    return float(np.clip(sum(contributions) / total_weight, -2.5, 2.5))


def _build_context_features(dataset: StockDataset, stock_returns: pd.Series) -> dict[str, Any]:
    aligned = pd.DataFrame({"stock": stock_returns})
    used_assets: list[str] = []

    for symbol, frame in dataset.context_price_history.items():
        if "Close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if len(close) < 120:
            continue
        aligned[symbol] = np.log(close).diff()
        used_assets.append(symbol)

    aligned = aligned.sort_index().fillna(0.0)
    macro = pd.DataFrame(index=stock_returns.index)
    for name, series in dataset.context_macro_series.items():
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            continue
        numeric.index = pd.to_datetime(numeric.index, errors="coerce")
        numeric = numeric[numeric.index.notna()].sort_index()
        macro[name] = numeric.reindex(stock_returns.index).ffill()

    raw_features = pd.DataFrame(index=stock_returns.index)
    raw_features["stock_momentum"] = stock_returns.rolling(20, min_periods=10).mean()
    raw_features["stock_reversal"] = -stock_returns.rolling(5, min_periods=3).sum()
    raw_features["stock_vol"] = stock_returns.rolling(20, min_periods=10).std()

    if "SPY" in aligned:
        raw_features["market_trend"] = aligned["SPY"].rolling(20, min_periods=10).mean()
        raw_features["market_stress"] = aligned["SPY"].rolling(20, min_periods=10).std()
    if {"IWM", "SPY"}.issubset(aligned.columns):
        raw_features["breadth"] = (aligned["IWM"] - aligned["SPY"]).rolling(10, min_periods=5).mean()
    if {"QQQ", "SPY"}.issubset(aligned.columns):
        raw_features["growth_leadership"] = (aligned["QQQ"] - aligned["SPY"]).rolling(10, min_periods=5).mean()
    if {"GLD", "SPY"}.issubset(aligned.columns):
        raw_features["safety_bid"] = (aligned["GLD"] - aligned["SPY"]).rolling(10, min_periods=5).mean()
    if {"UUP", "SPY"}.issubset(aligned.columns):
        raw_features["dollar_pressure"] = (aligned["UUP"] - aligned["SPY"]).rolling(10, min_periods=5).mean()
    if "USO" in aligned:
        raw_features["oil_shock"] = (
            aligned["USO"].rolling(10, min_periods=5).mean().abs()
            + aligned["USO"].rolling(20, min_periods=10).std()
        )
    if "TLT" in aligned:
        raw_features["rate_stress"] = (
            (-aligned["TLT"]).rolling(10, min_periods=5).mean()
            + aligned["TLT"].rolling(20, min_periods=10).std()
        )
    if "GPR" in macro:
        raw_features["geopolitical_risk_index"] = _zscore(macro["GPR"], lookback=126)
    if "GPR_Threat" in macro:
        raw_features["geopolitical_threat"] = _zscore(macro["GPR_Threat"], lookback=126)
    if "GPR_Act" in macro:
        raw_features["geopolitical_act"] = _zscore(macro["GPR_Act"], lookback=126)

    sector_etf = dataset.raw_info.get("sector_etf")
    if sector_etf and sector_etf in aligned.columns and "SPY" in aligned.columns:
        raw_features["sector_relative"] = (aligned[sector_etf] - aligned["SPY"]).rolling(15, min_periods=8).mean()

    for column in list(raw_features.columns):
        if column.startswith("geopolitical_"):
            continue
        raw_features[column] = _zscore(raw_features[column], lookback=126)

    standardized = raw_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest = standardized.iloc[-1] if not standardized.empty else pd.Series(dtype=float)

    opportunity_score = _weighted_feature_score(
        latest,
        {
            "stock_momentum": 0.28,
            "market_trend": 0.20,
            "breadth": 0.14,
            "growth_leadership": 0.12,
            "sector_relative": 0.16,
            "stock_reversal": 0.10,
            "geopolitical_risk_index": -0.08,
        },
    )
    stress_score = _weighted_feature_score(
        latest,
        {
            "stock_vol": 0.22,
            "market_stress": 0.18,
            "safety_bid": 0.18,
            "dollar_pressure": 0.14,
            "oil_shock": 0.16,
            "rate_stress": 0.12,
            "geopolitical_risk_index": 0.12,
            "geopolitical_act": 0.08,
        },
    )
    geopolitical_score = _weighted_feature_score(
        latest,
        {
            "safety_bid": 0.26,
            "dollar_pressure": 0.20,
            "oil_shock": 0.34,
            "rate_stress": 0.20,
            "geopolitical_risk_index": 0.34,
            "geopolitical_threat": 0.18,
            "geopolitical_act": 0.28,
        },
    )
    behavior_score = _weighted_feature_score(
        latest,
        {
            "stock_momentum": 0.50,
            "stock_reversal": 0.30,
            "breadth": 0.20,
            "geopolitical_risk_index": -0.10,
        },
    )

    feature_snapshot = (
        latest.sort_values(key=lambda series: series.abs(), ascending=False)
        .head(8)
        .rename("z_score")
        .to_frame()
    )
    if not feature_snapshot.empty:
        feature_snapshot["interpretation"] = np.where(
            feature_snapshot["z_score"] >= 0,
            "supports upside / risk-on",
            "supports downside / risk-off",
        )

    return {
        "features": standardized,
        "used_assets": used_assets,
        "feature_snapshot": feature_snapshot,
        "scores": {
            "opportunity": opportunity_score,
            "stress": stress_score,
            "geopolitical": geopolitical_score,
            "behavior": behavior_score,
        },
        "macro_available": list(dataset.context_macro_series),
    }


def _fit_context_mean_model(returns: pd.Series, features: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {"current_bias": 0.0, "feature_betas": {}, "driver_table": pd.DataFrame()}

    common = pd.concat(
        [
            returns.rename("y"),
            returns.shift(1).rename("lag1"),
            features.shift(1),
        ],
        axis=1,
    ).dropna()

    if len(common) < 120:
        return {"current_bias": 0.0, "feature_betas": {}, "driver_table": pd.DataFrame()}

    feature_columns = [column for column in common.columns if column not in {"y", "lag1"}]
    X = common[["lag1", *feature_columns]].to_numpy(dtype=float)
    y = common["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(X)), X])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)

    feature_betas = {
        column: float(np.clip(value, -0.006, 0.006))
        for column, value in zip(feature_columns, coefficients[2:])
    }
    latest_features = features.iloc[-1]
    contributions = {
        column: float(feature_betas[column] * latest_features.get(column, 0.0))
        for column in feature_columns
    }
    current_bias = float(np.clip(sum(contributions.values()), -0.012, 0.012))

    driver_table = pd.DataFrame(
        [
            {
                "feature": column,
                "beta": feature_betas[column],
                "current_z": float(latest_features.get(column, 0.0)),
                "drift_contribution": contributions[column],
            }
            for column in feature_columns
        ]
    )
    if not driver_table.empty:
        driver_table = driver_table.sort_values(
            "drift_contribution",
            key=lambda series: series.abs(),
            ascending=False,
        ).head(8)

    return {
        "current_bias": current_bias,
        "feature_betas": feature_betas,
        "driver_table": driver_table,
    }


def _tilt_probabilities(current_probabilities: np.ndarray, opportunity_score: float, stress_score: float) -> np.ndarray:
    adjusted = current_probabilities.astype(float).copy()
    if len(adjusted) >= 3:
        adjusted[0] *= np.exp(0.45 * stress_score - 0.22 * opportunity_score)
        adjusted[-1] *= np.exp(0.45 * opportunity_score - 0.22 * stress_score)
        adjusted[1] *= np.exp(-0.10 * abs(opportunity_score - stress_score))
    elif len(adjusted) == 2:
        adjusted[0] *= np.exp(0.40 * stress_score)
        adjusted[1] *= np.exp(0.40 * opportunity_score)

    adjusted = np.clip(adjusted, 1e-8, None)
    return adjusted / adjusted.sum()


def _tilt_transition_matrix(
    transition_matrix: np.ndarray,
    opportunity_score: float,
    stress_score: float,
) -> np.ndarray:
    tilted = transition_matrix.astype(float).copy()
    if tilted.shape[0] >= 3:
        tilted[0, :] *= np.exp(0.22 * stress_score - 0.08 * opportunity_score)
        tilted[-1, :] *= np.exp(0.22 * opportunity_score - 0.08 * stress_score)
        tilted[1, :] *= np.exp(-0.05 * abs(opportunity_score - stress_score))
    elif tilted.shape[0] == 2:
        tilted[0, :] *= np.exp(0.18 * stress_score)
        tilted[1, :] *= np.exp(0.18 * opportunity_score)

    tilted = np.clip(tilted, 1e-10, None)
    tilted = tilted / tilted.sum(axis=0, keepdims=True)
    return tilted


def _simulate_price_paths(
    current_price: float,
    current_probabilities: np.ndarray,
    transition_matrix: np.ndarray,
    regime_means: np.ndarray,
    regime_vol_scales: np.ndarray,
    base_sigma_path: np.ndarray,
    degrees_of_freedom: float,
    mean_model: dict[str, float],
    bootstrap_pool: np.ndarray,
    jump_process: dict[str, Any],
    context_model: dict[str, Any],
    context_scores: dict[str, float],
    simulations: int,
) -> np.ndarray:
    rng = np.random.default_rng(7)
    horizon = len(base_sigma_path)
    states = rng.choice(len(current_probabilities), size=simulations, p=current_probabilities)
    paths = np.empty((simulations, horizon + 1), dtype=float)
    paths[:, 0] = current_price
    previous_returns = np.full(simulations, float(mean_model.get("recent_mean", np.mean(regime_means))))

    scale = np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0)) if degrees_of_freedom > 2 else 1.0
    unconditional_mean = float(np.mean(regime_means))
    jump_probability = float(jump_process["jump_probability"])
    jump_pool = np.asarray(jump_process["jump_pool"], dtype=float)
    bootstrap_pool = np.asarray(bootstrap_pool, dtype=float)
    if len(bootstrap_pool) == 0:
        bootstrap_pool = np.array([0.0])

    context_bias = float(context_model["current_bias"])
    stress_score = float(context_scores.get("stress", 0.0))
    geopolitical_score = float(context_scores.get("geopolitical", 0.0))
    volatility_bias = float(
        np.clip(
            1.0 + 0.10 * max(stress_score, 0.0) + 0.05 * max(geopolitical_score, 0.0),
            0.85,
            1.45,
        )
    )
    jump_bias = float(
        np.clip(
            1.0 + 0.24 * max(stress_score, 0.0) + 0.18 * max(geopolitical_score, 0.0),
            0.85,
            1.80,
        )
    )

    for step in range(1, horizon + 1):
        decay = float(np.exp(-1.10 * (step - 1) / max(horizon - 1, 1)))
        states = _sample_next_states(rng, transition_matrix, states)
        regime_mu = regime_means[states]
        ar_mu = mean_model["intercept"] + mean_model["phi"] * previous_returns
        recent_mean = mean_model.get("recent_mean", unconditional_mean)
        contextual_mu = context_bias * decay
        mu = 0.32 * regime_mu + 0.24 * ar_mu + 0.18 * unconditional_mean + 0.16 * recent_mean + 0.10 * contextual_mu
        sigma = base_sigma_path[step - 1] * regime_vol_scales[states] * volatility_bias

        t_shocks = rng.standard_t(degrees_of_freedom, size=simulations) / scale
        empirical_shocks = rng.choice(bootstrap_pool, size=simulations, replace=True)
        innovation = 0.62 * t_shocks + 0.38 * empirical_shocks

        jump_flags = rng.random(simulations) < jump_probability * jump_bias
        jump_draws = rng.choice(jump_pool, size=simulations, replace=True)
        jumps = np.where(jump_flags, jump_draws, 0.0)

        log_returns = mu + sigma * innovation + jumps
        previous_returns = log_returns
        paths[:, step] = paths[:, step - 1] * np.exp(log_returns)

    return paths


def _build_horizon_summary(paths: np.ndarray, current_price: float) -> list[dict[str, Any]]:
    raw_horizons = [(1, "1 Day"), (5, "1 Week"), (10, "2 Weeks"), (21, "1 Month")]
    max_available = paths.shape[1] - 1
    horizons = [(days, label) for days, label in raw_horizons if days <= max_available]
    summaries: list[dict[str, Any]] = []

    for days, label in horizons:
        terminal_prices = paths[:, days]
        path_slice = paths[:, : days + 1]
        running_peaks = np.maximum.accumulate(path_slice, axis=1)
        drawdowns = path_slice / np.maximum(running_peaks, 1e-8) - 1.0
        max_drawdown = np.min(drawdowns, axis=1)
        summaries.append(
            {
                "label": label,
                "days": days,
                "mean": float(np.mean(terminal_prices)),
                "median": float(np.percentile(terminal_prices, 50)),
                "p05": float(np.percentile(terminal_prices, 5)),
                "p25": float(np.percentile(terminal_prices, 25)),
                "p75": float(np.percentile(terminal_prices, 75)),
                "p95": float(np.percentile(terminal_prices, 95)),
                "probability_upside": float(np.mean(terminal_prices > current_price)),
                "probability_down_5pct": float(np.mean(terminal_prices < current_price * 0.95)),
                "probability_up_5pct": float(np.mean(terminal_prices > current_price * 1.05)),
                "median_return": float(np.percentile(terminal_prices / current_price - 1.0, 50)),
                "expected_max_drawdown": float(np.mean(max_drawdown)),
                "p25_max_drawdown": float(np.percentile(max_drawdown, 25)),
                "p75_max_drawdown": float(np.percentile(max_drawdown, 75)),
            }
        )

    return summaries


def _build_recent_calibration(
    returns: pd.Series,
    regime_model: dict[str, Any],
    mean_model: dict[str, float],
    volatility_model: dict[str, Any],
) -> dict[str, Any]:
    filtered = regime_model.get("filtered_probabilities", regime_model["smoothed_probabilities"])
    regime_means = np.array([item["mean_return"] for item in regime_model["stats"]], dtype=float)
    regime_expected = pd.Series(
        filtered.to_numpy(dtype=float) @ regime_means,
        index=filtered.index,
        name="regime_expected",
    )
    ar_expected = pd.Series(
        mean_model["intercept"] + mean_model["phi"] * returns,
        index=returns.index,
        name="ar_expected",
    )
    long_mean = returns.expanding(min_periods=30).mean().rename("long_mean")
    sigma_series = pd.to_numeric(volatility_model["conditional_sigma"], errors="coerce").rename("sigma")
    realized_next = returns.shift(-1).rename("realized_next")
    predicted_mean = (
        0.42 * regime_expected + 0.33 * ar_expected + 0.25 * long_mean
    ).rename("predicted_mean")

    common = pd.concat([predicted_mean, sigma_series, realized_next], axis=1).dropna().tail(126)
    if common.empty:
        return {
            "sample_days": 0,
            "sign_accuracy": None,
            "interval_90_coverage": None,
            "left_tail_hit_rate": None,
            "predicted_daily_vol": None,
            "realized_daily_vol": None,
            "vol_ratio": None,
            "score": 0,
            "label": "Unavailable",
            "summary": "Δεν βγήκαν αρκετά observations για calibration diagnostics.",
        }

    residual_pool = np.asarray(volatility_model["std_resid"], dtype=float)
    if len(residual_pool) < 20:
        lower_q, upper_q = -1.64, 1.64
    else:
        lower_q, upper_q = np.quantile(residual_pool, [0.05, 0.95])

    lower_band = common["predicted_mean"] + common["sigma"] * lower_q
    upper_band = common["predicted_mean"] + common["sigma"] * upper_q

    sign_accuracy = float(
        np.mean(np.sign(common["predicted_mean"].to_numpy()) == np.sign(common["realized_next"].to_numpy()))
    )
    interval_coverage = float(
        np.mean(
            (common["realized_next"] >= lower_band)
            & (common["realized_next"] <= upper_band)
        )
    )
    left_tail_hit_rate = float(np.mean(common["realized_next"] < lower_band))
    predicted_daily_vol = float(common["sigma"].mean())
    realized_daily_vol = float(common["realized_next"].std(ddof=0))
    vol_ratio = realized_daily_vol / max(predicted_daily_vol, 1e-8)

    coverage_score = float(np.clip(1.0 - abs(interval_coverage - 0.90) / 0.20, 0.0, 1.0))
    sign_score = float(np.clip((sign_accuracy - 0.50) / 0.15, 0.0, 1.0))
    vol_score = float(np.clip(1.0 - abs(np.log(max(vol_ratio, 1e-8))) / 0.60, 0.0, 1.0))
    sample_score = float(np.clip(len(common) / 126.0, 0.0, 1.0))
    score = int(round(100.0 * (0.35 * coverage_score + 0.25 * sign_score + 0.25 * vol_score + 0.15 * sample_score)))

    if score >= 75:
        label = "Strong"
    elif score >= 55:
        label = "Balanced"
    else:
        label = "Fragile"

    summary = (
        "Το πρόσφατο calibration δείχνει καλή ισορροπία ανάμεσα σε direction, volatility και interval coverage."
        if label == "Strong"
        else "Το calibration είναι usable αλλά θέλει προσοχή, ειδικά όταν η αγορά αλλάζει regime."
        if label == "Balanced"
        else "Το μοντέλο παραμένει χρήσιμο σαν probability map, αλλά το πρόσφατο calibration είναι αδύναμο."
    )

    return {
        "sample_days": int(len(common)),
        "sign_accuracy": sign_accuracy,
        "interval_90_coverage": interval_coverage,
        "left_tail_hit_rate": left_tail_hit_rate,
        "predicted_daily_vol": predicted_daily_vol,
        "realized_daily_vol": realized_daily_vol,
        "vol_ratio": vol_ratio,
        "score": score,
        "label": label,
        "summary": summary,
    }


def _build_forecast_takeaways(
    dataset: StockDataset,
    horizon_summary: list[dict[str, Any]],
    regime_label: str,
    regime_probability: float,
    context_scores: dict[str, float],
    calibration: dict[str, Any],
) -> list[str]:
    takeaways: list[str] = []
    one_month = next((item for item in horizon_summary if item["label"] == "1 Month"), horizon_summary[-1])
    one_week = next((item for item in horizon_summary if item["label"] == "1 Week"), horizon_summary[0])

    takeaways.append(
        f"Το μοντέλο βλέπει πιο πιθανό regime το `{regime_label}` με πιθανότητα περίπου {regime_probability:.0%}."
    )

    if one_month["probability_upside"] >= 0.60:
        takeaways.append(
            f"Στο `1 Month` το probabilistic bias παραμένει ανοδικό, με median move περίπου {one_month['median_return']:.1%}."
        )
    elif one_month["probability_upside"] <= 0.40:
        takeaways.append(
            f"Στο `1 Month` το distribution γέρνει αμυντικά, με αυξημένη πιθανότητα downside tails."
        )
    else:
        takeaways.append(
            "Στο `1 Month` το distribution είναι σχετικά ισορροπημένο, άρα το signal είναι περισσότερο tactical παρά directional."
        )

    if max(context_scores.get("stress", 0.0), context_scores.get("geopolitical", 0.0)) >= 0.9:
        takeaways.append(
            "Τα stress / geopolitical proxies είναι ανεβασμένα, οπότε το μοντέλο βαραίνει περισσότερο volatility spikes και jump risk."
        )
    elif context_scores.get("opportunity", 0.0) >= 0.7 and one_week["probability_upside"] >= 0.55:
        takeaways.append(
            f"Το near-term setup παραμένει σχετικά risk-on για το `{dataset.symbol}`, ειδικά στο 1-week horizon."
        )
    else:
        takeaways.append(
            "Οι cross-asset proxies δεν δίνουν ακραίο καθεστώς, άρα το αποτέλεσμα εξαρτάται περισσότερο από το ίδιο το price regime της μετοχής."
        )

    calibration_label = calibration.get("label", "Unavailable")
    takeaways.append(
        f"Η πρόσφατη στατιστική αξιοπιστία του forecast engine αξιολογείται ως `{calibration_label}`."
    )
    return takeaways


def build_forecast_report(dataset: StockDataset, config: ForecastConfig) -> dict[str, Any]:
    close_prices = pd.to_numeric(dataset.price_history["Close"], errors="coerce").dropna()
    if len(close_prices) < 252:
        raise ForecastModelError("Δεν υπάρχουν αρκετά ιστορικά daily prices για forecasting.")

    log_returns = np.log(close_prices).diff().dropna()
    if len(log_returns) < 200:
        raise ForecastModelError("Τα returns είναι πολύ λίγα για σταθερό regime/volatility model.")

    try:
        regime_model = _fit_markov_model(log_returns, config.markov_regimes)
    except ForecastModelError:
        regime_model = _fallback_empirical_markov(log_returns, regimes=3)

    volatility_model = _fit_volatility_ensemble(log_returns, config.max_horizon)
    mean_model = _fit_autoregressive_mean_model(log_returns)
    jump_process = _estimate_jump_process(log_returns)
    context_features = _build_context_features(dataset, log_returns)
    context_model = _fit_context_mean_model(log_returns, context_features["features"])
    calibration = _build_recent_calibration(log_returns, regime_model, mean_model, volatility_model)

    overall_std = float(log_returns.std(ddof=0))
    regime_means = np.array([item["mean_return"] for item in regime_model["stats"]], dtype=float)
    regime_stds = np.array([item["std_return"] for item in regime_model["stats"]], dtype=float)
    regime_vol_scales = np.clip(regime_stds / max(overall_std, 1e-6), 0.65, 1.75)
    bootstrap_pool = volatility_model["std_resid"]

    adjusted_probabilities = _tilt_probabilities(
        regime_model["current_probabilities"],
        opportunity_score=context_features["scores"]["opportunity"],
        stress_score=context_features["scores"]["stress"],
    )
    adjusted_transition_matrix = _tilt_transition_matrix(
        regime_model["transition_matrix"],
        opportunity_score=context_features["scores"]["opportunity"],
        stress_score=context_features["scores"]["stress"],
    )

    simulated_paths = _simulate_price_paths(
        current_price=dataset.current_price,
        current_probabilities=adjusted_probabilities,
        transition_matrix=adjusted_transition_matrix,
        regime_means=regime_means,
        regime_vol_scales=regime_vol_scales,
        base_sigma_path=volatility_model["sigma_path"],
        degrees_of_freedom=volatility_model["degrees_of_freedom"],
        mean_model=mean_model,
        bootstrap_pool=bootstrap_pool,
        jump_process=jump_process,
        context_model=context_model,
        context_scores=context_features["scores"],
        simulations=config.simulations,
    )

    last_date = pd.to_datetime(close_prices.index[-1])
    forecast_dates = pd.bdate_range(last_date, periods=config.max_horizon + 1)
    fan_chart = pd.DataFrame(
        {
            "p05": np.percentile(simulated_paths, 5, axis=0),
            "p25": np.percentile(simulated_paths, 25, axis=0),
            "p50": np.percentile(simulated_paths, 50, axis=0),
            "p75": np.percentile(simulated_paths, 75, axis=0),
            "p95": np.percentile(simulated_paths, 95, axis=0),
        },
        index=forecast_dates,
    )

    current_regime_index = int(np.argmax(adjusted_probabilities))
    transition_df = pd.DataFrame(
        adjusted_transition_matrix,
        index=[f"To {label}" for label in regime_model["labels"]],
        columns=[f"From {label}" for label in regime_model["labels"]],
    )
    volatility_models = pd.DataFrame(volatility_model["models"])
    if not volatility_models.empty:
        volatility_models["weight"] = volatility_models["weight"].astype(float)

    horizon_summary = _build_horizon_summary(simulated_paths, dataset.current_price)
    takeaways = _build_forecast_takeaways(
        dataset=dataset,
        horizon_summary=horizon_summary,
        regime_label=regime_model["labels"][current_regime_index],
        regime_probability=float(adjusted_probabilities[current_regime_index]),
        context_scores=context_features["scores"],
        calibration=calibration,
    )

    return {
        "model_name": (
            f"{regime_model['model_name']} + AR(1) mean blend + "
            "GARCH/GJR-t volatility ensemble + bootstrap/jump Monte Carlo"
        ),
        "current_regime": regime_model["labels"][current_regime_index],
        "current_regime_probability": float(adjusted_probabilities[current_regime_index]),
        "horizon_summary": horizon_summary,
        "fan_chart": fan_chart,
        "paths": simulated_paths,
        "smoothed_probabilities": regime_model["smoothed_probabilities"],
        "transition_matrix": transition_df,
        "regime_stats": regime_model["stats"],
        "volatility_path": volatility_model["sigma_path"],
        "volatility_models": volatility_models,
        "selected_volatility_model": volatility_model["selected_model"],
        "mean_model": mean_model,
        "jump_process": jump_process,
        "context_scores": context_features["scores"],
        "context_assets_used": context_features["used_assets"],
        "macro_context_used": context_features["macro_available"],
        "feature_snapshot": context_features["feature_snapshot"],
        "context_driver_table": context_model["driver_table"],
        "context_bias": float(context_model["current_bias"]),
        "calibration": calibration,
        "takeaways": takeaways,
        "proxy_methodology": (
            "Geopolitical and behavior effects are approximated through free-market proxies such as "
            "oil, gold, dollar, duration, breadth, growth leadership, sector relative strength, realized "
            "market stress, and an open-access geopolitical risk index when available."
        ),
    }
