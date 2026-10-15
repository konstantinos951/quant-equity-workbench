"""Online supervised experts, trained only after the target return is observed.

Checkpoints accelerate exact-prefix replays; they never supply future-trained
parameters to historical validation. Simulation calls use frozen coefficients.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

VERSION = 1
AGENTS = ('Trend learner', 'Reversion learner', 'Autoregressive learner', 'Market learner', 'Random walk')
FEATURES = ((0, 2, 3), (0, 4), (0, 1, 2, 3), (0, 1, 5, 6))


class OnlineAgents:
    """Forgetting-factor recursive least squares and discounted expert losses."""

    def __init__(self) -> None:
        self.coefficients = [np.zeros(len(cols)) for cols in FEATURES]
        self.covariances = [np.eye(len(cols)) * .5 for cols in FEATURES]
        self.losses = np.ones(len(AGENTS))
        self.count = 0
        self.last = self.fast = self.slow = self.reversion = 0.
        self.market_last = self.market_mean = 0.
        self.variance = .0001

    def features(self) -> np.ndarray:
        scale = np.sqrt(max(self.variance, 1e-8))
        return np.array([1., *np.clip(np.array([
            self.last, self.fast, self.slow, self.reversion,
            self.market_last, self.market_mean,
        ]) / scale, -5, 5)])

    def weights(self) -> np.ndarray:
        logits = -4 * (self.losses - np.min(self.losses))
        weights = np.exp(np.clip(logits, -30, 0))
        weights /= weights.sum()
        # Exploration prevents an agent disappearing permanently after a regime change.
        return .95 * weights + .05 / len(AGENTS)

    def predict(self) -> tuple[np.ndarray, float]:
        x = self.features()
        sigma = np.sqrt(max(self.variance, 1e-8))
        forecasts = np.array([float(beta @ x[list(cols)]) * sigma
                              for beta, cols in zip(self.coefficients, FEATURES)] + [0.])
        forecasts = np.clip(forecasts, -.01, .01)
        return forecasts, float(forecasts @ self.weights())

    def observe(self, realized: float, market_return: float | None = None) -> None:
        """Score the issued forecast first, then learn from the observed target."""
        x = self.features()
        sigma = np.sqrt(max(self.variance, 1e-8))
        forecasts, _ = self.predict()
        losses = np.minimum(((realized - forecasts) / sigma) ** 2, 25)
        self.losses = .97 * self.losses + .03 * losses
        for i, cols in enumerate(FEATURES):
            features = x[list(cols)]
            covariance = self.covariances[i]
            projected = covariance @ features
            gain = projected / (.995 + features @ projected)
            error = np.clip(realized / sigma - self.coefficients[i] @ features, -3, 3)
            self.coefficients[i] += gain * error
            covariance = (covariance - np.outer(gain, projected)) / .995
            self.covariances[i] = .5 * (covariance + covariance.T)
        self.fast += (2 / 6) * (realized - self.fast)
        self.slow += (2 / 21) * (realized - self.slow)
        self.reversion = float(np.clip((59 / 60) * (self.reversion - realized / 60), -.01, .01))
        self.last = realized
        if market_return is not None and np.isfinite(market_return):
            self.market_last = market_return
            self.market_mean += (2 / 21) * (market_return - self.market_mean)
        else:
            self.market_last = 0.
            self.market_mean *= 19 / 21
        self.variance = max(.95 * self.variance + .05 * realized ** 2, 1e-8)
        self.count += 1

    def snapshot(self) -> dict[str, Any]:
        return {**{name: getattr(self, name) for name in (
            'count', 'last', 'fast', 'slow', 'reversion', 'market_last', 'market_mean', 'variance')},
            'coefficients': [x.tolist() for x in self.coefficients],
            'covariances': [x.tolist() for x in self.covariances], 'losses': self.losses.tolist()}

    @classmethod
    def restore(cls, data: dict[str, Any]) -> OnlineAgents:
        model = cls()
        for name in ('count', 'last', 'fast', 'slow', 'reversion', 'market_last', 'market_mean', 'variance'):
            setattr(model, name, data[name])
        model.coefficients = [np.asarray(x, dtype=float) for x in data['coefficients']]
        model.covariances = [np.asarray(x, dtype=float) for x in data['covariances']]
        model.losses = np.asarray(data['losses'], dtype=float)
        if (not isinstance(model.count, int) or model.count < 0 or model.variance <= 0
                or len(model.coefficients) != len(FEATURES) or len(model.covariances) != len(FEATURES)
                or model.losses.shape != (len(AGENTS),)):
            raise ValueError('Invalid online learning checkpoint.')
        for beta, covariance, cols in zip(model.coefficients, model.covariances, FEATURES):
            if (beta.shape != (len(cols),) or covariance.shape != (len(cols), len(cols))
                    or not np.isfinite(beta).all() or not np.isfinite(covariance).all()
                    or np.linalg.eigvalsh(covariance).min() <= 0):
                raise ValueError('Invalid online regression parameters.')
        scalars = [getattr(model, key) for key in ('last', 'fast', 'slow', 'reversion', 'market_last', 'market_mean', 'variance')]
        if not np.isfinite(scalars).all() or not np.isfinite(model.losses).all():
            raise ValueError('Nonfinite online learning checkpoint.')
        return model


def online_forecasts(returns: pd.Series, market: pd.Series | None,
                     checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Replay chronologically, resuming only an identical historical prefix.

    Revisions, changed windows or algorithm versions trigger a clean replay.
    Predictions stored at t were made before observing return t.
    """
    market_values = (market.reindex(returns.index).to_numpy(dtype=float)
                     if market is not None else np.full(len(returns), np.nan))
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all() or returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValueError('Online learning requires finite, uniquely dated, chronological returns.')
    data = pd.DataFrame({'stock': values, 'market': market_values}, index=returns.index)
    hashes = pd.util.hash_pandas_object(data, index=True).to_numpy().tobytes()

    def fingerprint(count: int) -> str:
        return hashlib.sha256(hashes[:count * 8]).hexdigest()

    model = OnlineAgents()
    forecasts: list[list[float]] = []
    status = 'replayed'
    if checkpoint_path is not None:
        try:
            saved = json.loads(checkpoint_path.read_text())
            count = saved['state']['count']
            if (saved['version'] == VERSION and isinstance(count, int) and 0 <= count <= len(values)
                    and saved['fingerprint'] == fingerprint(count)):
                restored = OnlineAgents.restore(saved['state'])
                history = np.asarray(saved['forecasts'], dtype=float)
                if history.shape != (count, 2) or not np.isfinite(history).all():
                    raise ValueError('Invalid checkpoint forecast history.')
                model, forecasts = restored, history.tolist()
                status = 'reused' if count == len(values) else 'resumed'
        except (OSError, ValueError, TypeError, KeyError, OverflowError, np.linalg.LinAlgError):
            pass
    previous_count = model.count
    for i in range(model.count, len(values)):
        agents, ensemble = model.predict()
        forecasts.append([ensemble, float(agents[3])])
        model.observe(float(values[i]), float(market_values[i]))
    warning = None
    if checkpoint_path is not None:
        temporary = None
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=checkpoint_path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({'version': VERSION, 'fingerprint': fingerprint(len(values)),
                           'state': model.snapshot(), 'forecasts': forecasts}, handle, allow_nan=False)
            os.replace(temporary, checkpoint_path)
        except (OSError, ValueError) as exc:
            warning = f'Learning checkpoint could not be saved ({type(exc).__name__}); forecasts were computed in memory.'
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
    agent_predictions, ensemble = model.predict()
    return {'predictions': pd.DataFrame(forecasts, index=returns.index, columns=['Learning agents', 'Dynamic ARX']),
            'state': model.snapshot(), 'next_drift': ensemble, 'next_arx_drift': float(agent_predictions[3]),
            'agents': pd.DataFrame({'agent': AGENTS, 'weight': model.weights(),
                                    'discounted_normalized_loss': model.losses, 'next_log_return': agent_predictions}),
            'observations': model.count, 'new_observations': model.count - previous_count,
            'checkpoint_status': status, 'warning': warning,
            'last_observation': str(returns.index[-1].date())}


def learning_path_drift(selection: dict[str, Any], previous_returns: np.ndarray,
                        step: int, state: dict[str, np.ndarray]) -> np.ndarray:
    """Propagate signals along simulated paths without training on fake outcomes."""
    learned = selection['learning']['state']
    if not state:
        state['last'] = np.full_like(previous_returns, learned['last'])
        state['fast'] = np.full_like(previous_returns, learned['fast'])
        state['slow'] = np.full_like(previous_returns, learned['slow'])
        state['reversion'] = np.full_like(previous_returns, learned['reversion'])
    if step > 0:
        state['last'] = previous_returns.copy()
        state['fast'] += (2 / 6) * (previous_returns - state['fast'])
        state['slow'] += (2 / 21) * (previous_returns - state['slow'])
        state['reversion'] = np.clip((59 / 60) * (state['reversion'] - previous_returns / 60), -.01, .01)
    sigma = np.sqrt(max(learned['variance'], 1e-8))
    # Unknown future market lags revert to zero; no market foresight is assumed.
    market_decay = np.exp(-step / 5)
    features = np.column_stack([np.ones_like(previous_returns),
        *[np.clip(state[key] / sigma, -5, 5) for key in ('last', 'fast', 'slow', 'reversion')],
        np.full_like(previous_returns, np.clip(learned['market_last'] * market_decay / sigma, -5, 5)),
        np.full_like(previous_returns, np.clip(learned['market_mean'] * market_decay / sigma, -5, 5))])
    forecasts = np.column_stack([
        np.clip(features[:, list(cols)] @ np.asarray(beta) * sigma, -.01, .01)
        for cols, beta in zip(FEATURES, learned['coefficients'])] + [np.zeros_like(previous_returns)])
    if selection['selected_model'] == 'Dynamic ARX':
        return forecasts[:, 3]
    return forecasts @ selection['learning']['agents']['weight'].to_numpy(dtype=float)
