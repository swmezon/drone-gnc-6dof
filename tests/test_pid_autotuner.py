import numpy as np

from src.pid_autotuner import (
    AutoTuneTargets,
    StrongPortfolioTargets,
)


def test_target_defaults_are_valid():
    targets = StrongPortfolioTargets()

    np.testing.assert_allclose(
        targets.post_transient_rmse_deg,
        np.array([0.25, 0.25, 0.50]),
    )
    np.testing.assert_allclose(
        targets.rmse_deg,
        np.array([0.25, 0.25, 0.50]),
    )


def test_backward_compatible_target_name():
    targets = AutoTuneTargets()

    np.testing.assert_allclose(
        targets.post_transient_rmse_deg,
        np.array([0.25, 0.25, 0.50]),
    )


def test_rmse_default_is_not_shared_between_instances():
    a = StrongPortfolioTargets()
    b = StrongPortfolioTargets()

    assert a.post_transient_rmse_deg is not b.post_transient_rmse_deg
