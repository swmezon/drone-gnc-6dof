from src.pid_autotuner import StrongPortfolioTargets


def test_target_vectors_have_three_axes():
    targets = StrongPortfolioTargets()
    assert targets.post_transient_rmse_deg.shape == (3,)
    assert targets.max_saturation_fraction.shape == (3,)
