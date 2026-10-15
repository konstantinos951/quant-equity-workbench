"""Causal daily drift models and chronological forecast comparisons.

The agent model is a stylized market with trend followers, mean reverters,
noise traders and adaptive strategy shares. Its fair value is a trailing-price
anchor, not independently observed intrinsic value.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from services.online_learning import online_forecasts, learning_path_drift

MODELS = ('Random walk', 'Historical drift', 'AR(1)', 'Ridge AR(5)', 'CAPM',
          'Adaptive agents', 'Empirical Markov', 'Learning agents', 'Dynamic ARX')


def _agent_state(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = pd.Series(values)
    momentum = history.ewm(span=20, adjust=False).mean().to_numpy()
    # log(price / trailing geometric mean), expressed in daily-return units.
    log_prices = np.cumsum(values)
    anchor = pd.Series(log_prices).ewm(alpha=1 / 60, adjust=False).mean().to_numpy()
    reversion = np.clip((anchor - log_prices) / 60, -.01, .01)
    signals = np.column_stack([momentum, reversion, np.zeros(len(values))])
    # At t, only strategies formed at t-1 earn the return observed at t.
    profits = np.zeros_like(signals)
    profits[1:] = signals[:-1] * values[1:, None]
    sigma_sq = max(float(np.var(values)), 1e-8)
    scores = pd.DataFrame(profits).ewm(span=40, adjust=False).mean().iloc[-1].to_numpy() / sigma_sq
    shares = np.exp(np.clip(4 * scores, -10, 10))
    shares /= shares.sum()
    return signals[-1], shares, scores


def _fit_models(returns: pd.Series, market: pd.Series | None, annual_rf: float) -> dict[str, Any]:
    values = returns.tail(504).to_numpy(dtype=float)
    lag = values[:-1]
    design = np.column_stack([np.ones(len(lag)), lag])
    coef = np.linalg.lstsq(design, values[1:], rcond=None)[0]
    # Shrink weak daily predictability rather than extrapolating large AR slopes.
    phi = float(np.clip(coef[1], -.35, .35))
    ridge_sigma = np.sqrt(max(float(np.var(values)), 1e-8))
    ridge_x = np.column_stack([np.ones(len(values) - 5),
                              *[values[5 - lag:len(values) - lag] / ridge_sigma for lag in range(1, 6)]])
    penalty = np.diag([0., 10., 10., 10., 10., 10.])
    ridge_coef = np.linalg.solve(ridge_x.T @ ridge_x + penalty, ridge_x.T @ (values[5:] / ridge_sigma))
    signals, shares, scores = _agent_state(values)
    capm = None
    if market is not None:
        aligned = pd.concat([returns.rename('stock'), market.rename('market')], axis=1).replace([np.inf, -np.inf], np.nan).dropna().tail(504)
        if len(aligned) >= 120:
            rf = float(np.expm1(np.log1p(annual_rf) / 252))
            stock = np.expm1(aligned['stock'].to_numpy()) - rf
            factor = np.expm1(aligned['market'].to_numpy()) - rf
            if np.var(factor) > 1e-10:
                result = sm.OLS(stock, sm.add_constant(factor)).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
                beta = float(result.params[1])
                expected_simple = rf + beta * float(np.mean(factor))
                total_variance = beta ** 2 * np.var(factor) + np.var(result.resid)
                capm = {'beta': beta, 'alpha_daily': float(result.params[0]),
                        'beta_hac_se': float(result.bse[1]), 'r_squared': float(result.rsquared),
                        'observations': len(aligned),
                        'daily_log_drift': float(np.clip(expected_simple - .5 * total_variance, -.01, .01))}
    # Causal discrete regimes, separate from the full hidden-state volatility fit.
    history = pd.Series(values)
    momentum = history.ewm(span=20, adjust=False).mean().to_numpy()
    sigma = history.ewm(span=20, adjust=False).std().fillna(0).to_numpy()
    signal = momentum / np.maximum(sigma, 1e-6)
    states = np.where(signal < -.15, 0, np.where(signal > .15, 2, 1))
    counts = np.ones((3, 3))
    np.add.at(counts, (states[1:], states[:-1]), 1)
    transition = counts / counts.sum(axis=0, keepdims=True)
    means = np.array([values[states == state].sum() / (np.sum(states == state) + 30) for state in range(3)])
    return {'mean': float(np.mean(values)), 'variance': max(float(np.var(values)), 1e-8), 'intercept': float(coef[0]), 'phi': phi,
            'last_return': float(values[-1]), 'agent_signals': signals, 'agent_shares': shares, 'agent_scores': scores,
            'ridge_coefficients': ridge_coef, 'ridge_sigma': ridge_sigma, 'last_five': values[-5:][::-1].copy(),
            'capm': capm, 'transition': transition, 'regime_means': means,
            'regime_probabilities': np.eye(3)[states[-1]]}


def model_drifts(fit: dict[str, Any]) -> dict[str, float]:
    probabilities = fit['transition'] @ fit['regime_probabilities']
    values = {'Random walk': 0., 'Historical drift': fit['mean'],
              'AR(1)': fit['intercept'] + fit['phi'] * fit['last_return'],
              'Ridge AR(5)': float(np.r_[1., fit['last_five'] / fit['ridge_sigma']] @ fit['ridge_coefficients'] * fit['ridge_sigma']),
              'Adaptive agents': float(fit['agent_signals'] @ fit['agent_shares']),
              'Empirical Markov': float(probabilities @ fit['regime_means'])}
    if fit['capm'] is not None:
        values['CAPM'] = fit['capm']['daily_log_drift']
    return {name: float(np.clip(value, -.01, .01)) for name, value in values.items()}


def fit_drift_selection(returns: pd.Series, market: pd.Series | None, annual_rf: float,
                        checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Select on an earlier block; evaluate on a distinct, later rolling block.

    Each one-step prediction is refitted using strictly earlier observations.
    Model identity is frozen before the evaluation block. This evaluates drift,
    not the complete Monte Carlo distribution or multi-day interval coverage.
    """
    if not np.isfinite(annual_rf) or annual_rf <= -1:
        raise ValueError('Risk-free rate must be finite and greater than -1.')
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(returns) < 240 or returns.index.has_duplicates:
        raise ValueError('Drift selection needs at least 240 uniquely dated returns.')
    learning = online_forecasts(returns, market, checkpoint_path)
    block = min(63, (len(returns) - 160) // 2)
    rows = []
    for end in range(len(returns) - 2 * block, len(returns)):
        fit = _fit_models(returns.iloc[:end], market, annual_rf)
        sigma = float(returns.iloc[:end].ewm(span=40, adjust=False).std().iloc[-1])
        rows.append({**model_drifts(fit), **learning['predictions'].iloc[end].to_dict(),
                     'actual': float(returns.iloc[end]), 'sigma': max(sigma, 1e-8)})
    predictions = pd.DataFrame(rows, index=returns.index[-2 * block:])
    selection, evaluation = predictions.iloc[:block], predictions.iloc[block:]
    eligible = [name for name in MODELS if name in selection and selection[name].notna().all()]
    loss = {name: float(np.mean((selection[name] - selection.actual) ** 2)) for name in eligible}
    selected = min(loss, key=loss.get)
    # A small validation gain is insufficient evidence to displace the baseline.
    if loss[selected] >= .98 * loss['Random walk']:
        selected = 'Random walk'
    baseline_mse = float(np.mean(evaluation.actual ** 2))
    metrics = []
    for name in eligible:
        error = evaluation[name] - evaluation.actual
        mse = float(np.mean(error ** 2))
        metrics.append({'model': name, 'selected': name == selected,
                        'validation_rmse': np.sqrt(loss[name]), 'holdout_rmse': np.sqrt(mse),
                        'holdout_mae': float(np.mean(np.abs(error))),
                        'holdout_direction_accuracy': (float(np.mean(np.sign(evaluation[name]) == np.sign(evaluation.actual))) if name != 'Random walk' else np.nan),
                        'holdout_mse_skill_vs_random_walk': (1 - mse / baseline_mse if baseline_mse > 0 else np.nan)})
    fit = _fit_models(returns, market, annual_rf)
    live_drifts = {**model_drifts(fit), 'Learning agents': learning['next_drift'], 'Dynamic ARX': learning['next_arx_drift']}
    # Calibrate each interval using only errors whose outcomes are already known.
    interval_rows = []
    for offset in range(block, 2 * block):
        past = predictions.iloc[:offset].tail(126)
        errors = np.abs((past.actual - past[selected]) / past.sigma).to_numpy()
        rank = min(len(errors), int(np.ceil((len(errors) + 1) * .9)))
        quantile = float(np.sort(errors)[rank - 1])
        row = predictions.iloc[offset]
        interval_rows.append((row[selected] - quantile * row.sigma, row[selected] + quantile * row.sigma))
    holdout = evaluation[[selected, 'actual', 'sigma']].rename(columns={selected: 'predicted_mean', 'actual': 'realized_next'}).copy()
    holdout[['lower_band', 'upper_band']] = interval_rows
    differential = evaluation.actual ** 2 - (evaluation[selected] - evaluation.actual) ** 2
    gain_fit = sm.OLS(differential.to_numpy(), np.ones((len(differential), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': min(5, block - 1)})
    gain_ci = gain_fit.conf_int(alpha=.05)[0]
    evidence = {'mean_mse_gain': float(differential.mean()), 'hac_ci95_low': float(gain_ci[0]),
                'hac_ci95_high': float(gain_ci[1]),
                'positive_gain_supported': bool(gain_ci[0] > 0),
                'note': 'HAC 95% interval for mean squared-error reduction versus random walk on the later holdout. '
                        'Short samples and repeated model research limit inference; this does not validate trading profitability.'}
    return {'selected_model': selected, 'fit': fit, 'metrics': pd.DataFrame(metrics),
            'holdout_predictions': holdout, 'learning': learning, 'evidence': evidence,
            'validation_days': block, 'holdout_days': block,
            'holdout_start': str(evaluation.index[0].date()), 'holdout_end': str(evaluation.index[-1].date()),
            'daily_log_drift': live_drifts[selected],
            'methodology': 'Expanding-origin one-day log-return forecasts. Batch models use up to 504 training observations; '
            'online agents learn sequentially from the available history with forgetting. '
            'Model selected on the earlier validation block, requiring a 2% MSE gain over the zero-log-drift random walk. '
            'The later holdout is used only for reporting. CAPM uses historical market-premium estimates and a constant current risk-free assumption. '
            'One-day bands use prior standardized forecast errors with sequential conformal-style calibration; '
            'serial dependence can invalidate nominal coverage guarantees. '
            'Agent behavior is a stylized hypothesis, not observed investor activity. These scores do not validate Monte Carlo intervals or multi-day forecasts.'}


def simulate_drift(selection: dict[str, Any], previous_returns: np.ndarray,
                   cumulative_log_return: np.ndarray, step: int,
                   agent_state: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Conditional drift for each simulation, updated from simulated prices."""
    fit, name = selection['fit'], selection['selected_model']
    if name in ('Learning agents', 'Dynamic ARX'):
        return learning_path_drift(selection, previous_returns, step, agent_state if agent_state is not None else {})
    if name == 'Ridge AR(5)':
        state = agent_state if agent_state is not None else {}
        if 'lags' not in state:
            state['lags'] = np.tile(fit['last_five'], (len(previous_returns), 1))
        if step > 0:
            state['lags'][:, 1:] = state['lags'][:, :-1].copy()
            state['lags'][:, 0] = previous_returns
        design = np.column_stack([np.ones(len(previous_returns)), state['lags'] / fit['ridge_sigma']])
        return np.clip(design @ fit['ridge_coefficients'] * fit['ridge_sigma'], -.01, .01)
    if name == 'AR(1)':
        return np.clip(fit['intercept'] + fit['phi'] * previous_returns, -.01, .01)
    if name == 'Adaptive agents':
        state = agent_state if agent_state is not None else {}
        if not state:
            state['signals'] = np.tile(fit['agent_signals'], (len(previous_returns), 1))
            state['scores'] = np.tile(fit['agent_scores'], (len(previous_returns), 1))
        if step > 0:
            # Reward the previous signals before updating them with the new return.
            profits = state['signals'] * previous_returns[:, None] / fit['variance']
            state['scores'] += (2 / 41) * (profits - state['scores'])
            state['signals'][:, 0] += (2 / 21) * (previous_returns - state['signals'][:, 0])
            state['signals'][:, 1] = np.clip((59 / 60) * (state['signals'][:, 1] - previous_returns / 60), -.01, .01)
        logits = np.clip(4 * state['scores'], -10, 10)
        logits -= logits.max(axis=1, keepdims=True)
        shares = np.exp(logits)
        shares /= shares.sum(axis=1, keepdims=True)
        return np.clip(np.sum(shares * state['signals'], axis=1), -.01, .01)
    if name == 'Empirical Markov':
        probabilities = np.linalg.matrix_power(fit['transition'], step + 1) @ fit['regime_probabilities']
        return np.full_like(previous_returns, float(probabilities @ fit['regime_means']))
    return np.full_like(previous_returns, selection['daily_log_drift'])
