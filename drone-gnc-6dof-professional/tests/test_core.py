import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from drone_gnc.vehicles import QuadrotorModel
from drone_gnc.vehicles.quadrotor import rotation_matrix
from drone_gnc.control.cascaded import CascadedController
from drone_gnc.guidance.waypoints import WaypointGuidance
from drone_gnc.estimation.error_state_ekf import ErrorStateEKF15

def test_rotation_orthonormal():
    R=rotation_matrix(.2,-.1,.4); assert np.allclose(R.T@R,np.eye(3),atol=1e-10)
def test_hover_equilibrium():
    v=QuadrotorModel(); dx=v.derivatives(np.zeros(12),v.nominal_control()); assert np.linalg.norm(dx)<1e-10
def test_controller_outputs_finite():
    v=QuadrotorModel();c=CascadedController();u,_=c.command(np.zeros(12),np.array([0,0,1]),np.zeros(3),0,.01);assert np.isfinite(u).all() and 0<=u[0]<=v.params.max_thrust
def test_guidance_advances():
    g=WaypointGuidance(np.array([[0,0,0],[1,0,0]]),.2,1);_,_,_,idx,_=g.update(np.zeros(3));assert idx==1
def test_ekf_covariance_symmetric():
    e=ErrorStateEKF15(.01);e.predict(np.array([0,0,9.81]),np.zeros(3));assert np.allclose(e.P,e.P.T,atol=1e-12)
