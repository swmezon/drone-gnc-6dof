import numpy as np
from src.attitude_controller import CascadedAttitudeController, nominal_attitude_control_gains
from src.closed_loop_simulation import simulate_closed_loop_attitude
from src.parameters import nominal_parameters

def test_roll_step_converges():
    params = nominal_parameters()
    c = CascadedAttitudeController(nominal_attitude_control_gains())
    cmd = np.deg2rad(np.array([5.0,0.0,0.0]))

    result = simulate_closed_loop_attitude(
        np.zeros(12), lambda t: cmd, c, params, t_final=2.0, dt=0.005
    )

    final_roll_deg = np.rad2deg(result.state[-1,6])
    assert 4.0 < final_roll_deg < 6.0
