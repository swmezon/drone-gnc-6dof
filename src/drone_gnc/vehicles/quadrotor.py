from dataclasses import dataclass
import numpy as np
from .base import VehicleModel


def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    cph,sph=np.cos(phi),np.sin(phi); cth,sth=np.cos(theta),np.sin(theta); cps,sps=np.cos(psi),np.sin(psi)
    return np.array([
        [cps*cth, cps*sth*sph-sps*cph, cps*sth*cph+sps*sph],
        [sps*cth, sps*sth*sph+cps*cph, sps*sth*cph-cps*sph],
        [-sth, cth*sph, cth*cph],
    ])


def euler_rate_matrix(phi: float, theta: float) -> np.ndarray:
    cph,sph=np.cos(phi),np.sin(phi); cth=np.cos(theta)
    cth=np.sign(cth)*max(abs(cth),1e-6)
    tth=np.sin(theta)/cth
    return np.array([[1.0, sph*tth, cph*tth],[0.0,cph,-sph],[0.0,sph/cth,cph/cth]])

@dataclass
class QuadrotorParams:
    mass: float = 1.5
    gravity: float = 9.81
    inertia_x: float = 0.030
    inertia_y: float = 0.030
    inertia_z: float = 0.050
    linear_drag: float = 0.16
    angular_drag: float = 0.025
    max_tilt_deg: float = 30.0
    max_thrust: float = 30.0
    max_torque: float = 1.2

class QuadrotorModel(VehicleModel):
    """12-state nonlinear Newton-Euler quadrotor model.

    State: [x,y,z, u,v,w, phi,theta,psi, p,q,r]
    Coordinates use an inertial z-up frame.
    Control: [T, tau_x, tau_y, tau_z]
    """
    state_size=12; control_size=4
    def __init__(self, params: QuadrotorParams | None = None):
        self.params=params or QuadrotorParams()
        self.I=np.diag([self.params.inertia_x,self.params.inertia_y,self.params.inertia_z])
        self.Iinv=np.linalg.inv(self.I)

    def nominal_control(self) -> np.ndarray:
        return np.array([self.params.mass*self.params.gravity,0.,0.,0.])

    def derivatives(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        p=self.params
        pos=state[0:3]; vel=state[3:6]; ang=state[6:9]; omega=state[9:12]
        T=float(np.clip(control[0],0,p.max_thrust))
        tau=np.clip(control[1:4],-p.max_torque,p.max_torque)
        R=rotation_matrix(*ang)
        thrust_i=R@np.array([0.,0.,T])
        gravity=np.array([0.,0.,-p.mass*p.gravity])
        drag=-p.linear_drag*vel
        acc=(thrust_i+gravity+drag)/p.mass
        euler_dot=euler_rate_matrix(ang[0],ang[1])@omega
        omega_dot=self.Iinv@(tau-np.cross(omega,self.I@omega)-p.angular_drag*omega)
        return np.concatenate([vel,acc,euler_dot,omega_dot])
