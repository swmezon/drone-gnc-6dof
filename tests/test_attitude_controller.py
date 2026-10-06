import numpy as np
from src.attitude_controller import CascadedAttitudeController, nominal_attitude_control_gains

def test_zero_error_produces_zero_torque():
    c = CascadedAttitudeController(nominal_attitude_control_gains())
    torque, desired, attitude_error, rate_error = c.update(
        np.zeros(3), np.zeros(3), np.zeros(3), 0.005
    )
    np.testing.assert_allclose(torque, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(desired, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(attitude_error, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(rate_error, np.zeros(3), atol=1e-12)

def test_positive_roll_error_produces_positive_roll_torque():
    c = CascadedAttitudeController(nominal_attitude_control_gains())
    torque, *_ = c.update(
        np.deg2rad(np.array([10.0,0.0,0.0])),
        np.zeros(3),
        np.zeros(3),
        0.005,
    )
    assert torque[0] > 0.0
