import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from services.online_learning import OnlineAgents, online_forecasts, learning_path_drift
from services.econometrics import fit_drift_selection
from services.forecasting import _build_horizon_summary, _simulate_price_paths


class OnlineLearningTests(unittest.TestCase):
    def history(self, count=360):
        rng = np.random.default_rng(13)
        values = np.zeros(count)
        for i in range(1, count):
            values[i] = .75 * values[i - 1] + rng.normal(0, .002)
        return pd.Series(values, index=pd.bdate_range('2020-01-01', periods=count))

    def test_learns_predictable_structure(self):
        returns = self.history(1100)
        learned = online_forecasts(returns, None)
        predictions = learned['predictions']['Learning agents'].iloc[-300:]
        baseline = np.mean(returns.iloc[-300:] ** 2)
        skill = 1 - np.mean((predictions - returns.iloc[-300:]) ** 2) / baseline
        self.assertGreater(skill, .2)
        self.assertAlmostEqual(learned['agents'].weight.sum(), 1)
        self.assertTrue((learned['agents'].weight >= .01).all())
        self.assertGreater(learned['agents'].weight.max() - learned['agents'].weight.min(), .05)

    def test_predictions_precede_targets(self):
        returns = self.history()
        market = returns * .5
        original = online_forecasts(returns, market)
        changed = returns.copy()
        changed.iloc[210:] += .1
        changed_market = market.copy()
        changed_market.iloc[210:] -= .2
        altered = online_forecasts(changed, changed_market)
        # The prediction for day 210 must not see even that day's target or market move.
        np.testing.assert_allclose(original['predictions'].iloc[:211], altered['predictions'].iloc[:211])
        self.assertFalse(np.allclose(original['predictions'].iloc[211:], altered['predictions'].iloc[211:]))

    def test_checkpoint_resume_is_idempotent_and_matches_replay(self):
        returns = self.history()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agents.json'
            initial = online_forecasts(returns.iloc[:230], None, path)
            self.assertEqual(initial['new_observations'], 230)
            resumed = online_forecasts(returns, None, path)
            self.assertEqual(resumed['checkpoint_status'], 'resumed')
            self.assertEqual(resumed['new_observations'], 130)
            repeated = online_forecasts(returns, None, path)
            self.assertEqual(repeated['checkpoint_status'], 'reused')
            self.assertEqual(repeated['new_observations'], 0)
            fresh = online_forecasts(returns, None)
            self.assertEqual(resumed['state'], fresh['state'])
            self.assertEqual(repeated['state'], fresh['state'])
            np.testing.assert_allclose(resumed['predictions'], fresh['predictions'])
            # A checkpoint trained in the future cannot be used for an earlier cutoff.
            earlier = online_forecasts(returns.iloc[:250], None, path)
            self.assertEqual(earlier['checkpoint_status'], 'replayed')
            self.assertEqual(earlier['state'], online_forecasts(returns.iloc[:250], None)['state'])

    def test_revised_history_and_corruption_rebuild_checkpoint(self):
        returns = self.history()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agents.json'
            online_forecasts(returns, None, path)
            revised = returns.copy()
            revised.iloc[35] += .002
            replayed = online_forecasts(revised, None, path)
            self.assertEqual(replayed['checkpoint_status'], 'replayed')
            self.assertEqual(replayed['state'], online_forecasts(revised, None)['state'])
            payload = json.loads(path.read_text())
            payload['state']['losses'] = [float('nan')] * 5
            path.write_text(json.dumps(payload))
            repaired = online_forecasts(revised, None, path)
            self.assertEqual(repaired['checkpoint_status'], 'replayed')
            self.assertEqual(repaired['state'], replayed['state'])
            path.write_text('{broken')
            repaired_again = online_forecasts(revised, None, path)
            self.assertEqual(repaired_again['state'], replayed['state'])

    def test_read_only_checkpoint_does_not_prevent_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / 'not-a-directory'
            parent.write_text('occupied')
            result = online_forecasts(self.history(), None, parent / 'agents.json')
            self.assertIsNotNone(result['warning'])
            self.assertTrue(np.isfinite(result['next_drift']))

    def test_simulation_does_not_train_saved_agents(self):
        learned = online_forecasts(self.history(), None)
        snapshot = copy.deepcopy(learned['state'])
        selection = {'selected_model': 'Learning agents', 'learning': learned}
        state = {}
        previous = np.zeros(20)
        first = learning_path_drift(selection, previous, 0, state)
        np.testing.assert_allclose(first, learned['next_drift'])
        next_drift = learning_path_drift(selection, np.linspace(-.03, .03, 20), 1, state)
        self.assertGreater(float(np.ptp(next_drift)), 0)
        self.assertEqual(snapshot, learned['state'])
        self.assertTrue((np.abs(next_drift) <= .01).all())

    def test_checkpoint_cannot_leak_into_holdout(self):
        returns = self.history()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agents.json'
            original = fit_drift_selection(returns, None, .04, path)
            changed = returns.copy()
            changed.iloc[-original['holdout_days']:] *= -4
            altered = fit_drift_selection(changed, None, .04, path)
            self.assertEqual(original['selected_model'], altered['selected_model'])
            np.testing.assert_allclose(original['metrics'].validation_rmse, altered['metrics'].validation_rmse)
            # The first holdout interval also cannot use that day's changed outcome.
            np.testing.assert_allclose(original['holdout_predictions'].iloc[0][['predicted_mean', 'lower_band', 'upper_band']],
                                       altered['holdout_predictions'].iloc[0][['predicted_mean', 'lower_band', 'upper_band']])

    def test_tail_risk_and_probability_sampling_intervals(self):
        terminal = np.arange(60., 141.)
        paths = np.column_stack([np.full(len(terminal), 100.), terminal])
        result = _build_horizon_summary(paths, 100.)[0]
        returns = terminal / 100 - 1
        expected_var = -np.quantile(returns, .05)
        expected_es = -returns[returns <= -expected_var].mean()
        self.assertAlmostEqual(result['var_95_loss'], expected_var)
        self.assertAlmostEqual(result['expected_shortfall_95_loss'], expected_es)
        self.assertGreaterEqual(result['expected_shortfall_95_loss'], result['var_95_loss'])
        self.assertLess(result['probability_upside_mc_ci95_low'], result['probability_upside'])
        self.assertGreater(result['probability_upside_mc_ci95_high'], result['probability_upside'])

    def test_new_candidates_drive_monte_carlo_without_retraining(self):
        selection = fit_drift_selection(self.history(), None, .04)
        snapshot = copy.deepcopy(selection['learning']['state'])
        for name in ('Learning agents', 'Dynamic ARX', 'Ridge AR(5)'):
            with self.subTest(model=name):
                selection['selected_model'] = name
                paths = _simulate_price_paths(
                    current_price=100., current_probabilities=np.array([1.]), transition_matrix=np.ones((1, 1)),
                    regime_means=np.array([0.]), regime_vol_scales=np.ones(1), base_sigma_path=np.full(21, .02),
                    degrees_of_freedom=8., mean_model={'intercept': 0, 'phi': 0},
                    bootstrap_pool=np.array([-1., 0., 1.]), jump_process={'jump_probability': 0., 'jump_pool': np.array([0.])},
                    context_model={'current_bias': 0.}, context_scores={}, simulations=50, drift_selection=selection)
                self.assertEqual(paths.shape, (50, 22))
                self.assertTrue(np.isfinite(paths).all() and (paths > 0).all())
                self.assertEqual(snapshot, selection['learning']['state'])

    def test_observation_changes_parameters_and_prediction(self):
        model = OnlineAgents()
        before = copy.deepcopy(model.snapshot())
        for _ in range(30):
            model.observe(.003, .001)
        self.assertNotEqual(model.snapshot()['coefficients'], before['coefficients'])
        self.assertGreater(model.predict()[1], 0)
        self.assertEqual(model.count, 30)


if __name__ == '__main__':
    unittest.main()
