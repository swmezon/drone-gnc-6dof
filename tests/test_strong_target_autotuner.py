import numpy as np

from src.pid_autotuner import StrongPortfolioTargets


def test_strong_target_defaults():
    targets = StrongPortfolioTargets()

    np.testing.assert_allclose(
        targets.post_transient_rmse_deg,
        np.array([0.25, 0.25, 0.50]),
    )
    np.testing.assert_allclose(
        targets.max_saturation_fraction,
        np.array([0.02, 0.02, 0.05]),
    )

    assert targets.steady_state_error_deg == 0.10
    assert targets.percent_overshoot == 5.0
    assert targets.settling_time_s == 1.0
    assert targets.rise_time_min_s == 0.30
    assert targets.rise_time_max_s == 0.70
