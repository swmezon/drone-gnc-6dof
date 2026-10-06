from scripts.run_pid_autotune import CONTROLLER_HZ as AUTOTUNE_HZ
from scripts.run_closed_loop_attitude import CONTROLLER_HZ as VALIDATION_HZ, DT


def test_autotuner_runs_at_1000_hz():
    assert AUTOTUNE_HZ == 1000.0


def test_validation_runs_at_1000_hz():
    assert VALIDATION_HZ == 1000.0
    assert DT == 0.001
