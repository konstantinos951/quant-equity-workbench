import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from services.econometrics import _fit_models, fit_drift_selection, model_drifts, simulate_drift
from services.forecasting import _simulate_price_paths, build_forecast_report
from services.models import ForecastConfig


class EconometricsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(34)
        self.index = pd.bdate_range('2020-01-01', periods=360)
        self.market = pd.Series(rng.normal(.0004, .012, 360), index=self.index)
        rf = np.expm1(np.log1p(.04) / 252)
        stock_simple = rf + 1.6 * (np.expm1(self.market) - rf) + rng.normal(0, .002, 360)
        self.stock = np.log1p(stock_simple)

    def test_capm_beta_and_market_alignment(self):
        fit = _fit_models(self.stock, self.market, .04)
        self.assertAlmostEqual(fit['capm']['beta'], 1.6, delta=.06)
        self.assertGreater(fit['capm']['beta_hac_se'], 0)
        self.assertGreater(fit['capm']['r_squared'], .9)
        # Future market observations cannot enter a historical stock fit.
        prefix = self.stock.iloc[:200]
        changed = self.market.copy()
        changed.iloc[200:] = 2
        self.assertEqual(_fit_models(prefix, self.market, .04)['capm'], _fit_models(prefix, changed, .04)['capm'])

    def test_holdout_cannot_select_model(self):
        original = fit_drift_selection(self.stock, self.market, .04)
        changed = self.stock.copy()
        changed.iloc[-original['holdout_days']:] *= -3
        altered = fit_drift_selection(changed, self.market, .04)
        self.assertEqual(original['selected_model'], altered['selected_model'])
        np.testing.assert_allclose(original['metrics'].validation_rmse, altered['metrics'].validation_rmse)
        self.assertTrue((original['holdout_predictions'].index >= pd.Timestamp(original['holdout_start'])).all())

    def test_missing_market_and_constant_history(self):
        result = fit_drift_selection(self.stock, None, .04)
        self.assertNotIn('CAPM', result['metrics'].model.to_list())
        flat = fit_drift_selection(self.stock * 0, None, .04)
        self.assertEqual(flat['selected_model'], 'Random walk')
        self.assertEqual(flat['daily_log_drift'], 0)
        self.assertTrue(np.isfinite(flat['metrics'].holdout_rmse).all())

    def test_agent_forecast_and_transition_probabilities(self):
        fit = _fit_models(self.stock, self.market, .04)
        np.testing.assert_allclose(fit['transition'].sum(axis=0), 1)
        self.assertAlmostEqual(fit['agent_shares'].sum(), 1)
        selection = {'fit': fit, 'selected_model': 'Adaptive agents', 'daily_log_drift': model_drifts(fit)['Adaptive agents']}
        initial = simulate_drift(selection, np.zeros(10), np.zeros(10), 0)
        np.testing.assert_allclose(initial, selection['daily_log_drift'])
        stressed = simulate_drift(selection, np.full(10, -.1), np.full(10, -1.), 10)
        self.assertTrue(np.isfinite(stressed).all())
        self.assertTrue((np.abs(stressed) <= .01).all())

    def test_monte_carlo_shocks_keep_variance(self):
        rng = np.random.default_rng(5)
        selection = {'fit': {'last_return': 0}, 'selected_model': 'Random walk', 'daily_log_drift': 0}
        paths = _simulate_price_paths(
            current_price=100., current_probabilities=np.array([1.]), transition_matrix=np.ones((1, 1)),
            regime_means=np.array([0.]), regime_vol_scales=np.ones(1), base_sigma_path=np.array([.02]),
            degrees_of_freedom=8., mean_model={'intercept': 0, 'phi': 0},
            bootstrap_pool=rng.normal(0, 1, 10000), jump_process={'jump_probability': 0., 'jump_pool': np.array([0.])},
            context_model={'current_bias': 0.}, context_scores={}, simulations=50000, drift_selection=selection)
        realized = np.log(paths[:, 1] / paths[:, 0])
        self.assertAlmostEqual(float(realized.std()), .02, delta=.0005)
        self.assertAlmostEqual(float(realized.mean()), 0., delta=.0004)
        self.assertTrue((paths > 0).all())

    def test_complete_forecast_with_market_context(self):
        prices = 100 * np.exp(self.stock.cumsum())
        dataset = SimpleNamespace(
            symbol='TEST', current_price=float(prices.iloc[-1]), risk_free_rate=.04,
            price_history=pd.DataFrame({'Close': prices}),
            context_price_history={'SPY': pd.DataFrame({'Close': 100 * np.exp(self.market.cumsum())})},
            context_macro_series={}, raw_info={},
        )
        report = build_forecast_report(dataset, ForecastConfig(simulations=100))
        self.assertIsNotNone(report['drift_selection']['fit']['capm'])
        self.assertEqual(report['paths'].shape, (100, 22))
        self.assertTrue(np.isfinite(report['paths']).all())
        np.testing.assert_allclose(report['paths'][:, 0], dataset.current_price)
        self.assertEqual(report['calibration']['sample_days'], report['drift_selection']['holdout_days'])
        fan = report['fan_chart']
        self.assertTrue((fan.p05 <= fan.p50).all() and (fan.p50 <= fan.p95).all())


if __name__ == '__main__':
    unittest.main()
