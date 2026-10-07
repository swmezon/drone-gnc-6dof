import numpy as np
from src.aircraft import nominal_aircraft
from src.trim import trim_state_control
from src.linearize import linearize
from src.stability import local_rate_loop_analysis
from src.experiment import dispersed_aircraft


def test_trim_converges():
    p = nominal_aircraft()
    _, _, sol = trim_state_control(p)
    assert sol.success
    assert np.linalg.norm(sol.fun) < 1e-5


def test_linearization_dimensions():
    p = nominal_aircraft()
    x, u, _ = trim_state_control(p)
    A, B = linearize(x, u, p)
    assert A.shape == (12, 12)
    assert B.shape == (12, 4)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))


def test_rate_loop_metrics_are_finite():
    p = nominal_aircraft()
    x, u, _ = trim_state_control(p)
    A, B = linearize(x, u, p)
    out = local_rate_loop_analysis(A[9, 9], B[9, 0], 0.20)
    assert np.isfinite(out["gain_margin_db"])
    assert np.isfinite(out["phase_margin_deg"])
    assert np.isfinite(out["gain_cross_rad_s"])
    assert np.isfinite(out["closed_loop_bandwidth_rad_s"])


def test_aircraft_dispersion_changes_parameters():
    rng = np.random.default_rng(42)
    p = nominal_aircraft()
    pd = dispersed_aircraft(rng)
    assert pd.mass != p.mass
    assert pd.Jy != p.Jy
