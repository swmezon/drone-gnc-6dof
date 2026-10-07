from dataclasses import dataclass
import numpy as np

@dataclass
class AutopilotState:
    int_p: float=0.0
    int_q: float=0.0
    int_v: float=0.0
    prev_surfaces: np.ndarray=None
    def __post_init__(self):
        if self.prev_surfaces is None:
            self.prev_surfaces=np.zeros(3)


def wrap_pi(a):
    return (a+np.pi)%(2*np.pi)-np.pi

def guidance_command(pos_ned, wp_ned, chi, max_bank=np.deg2rad(30.0)):
    d=wp_ned[:2]-pos_ned[:2]
    chi_d=np.arctan2(d[1],d[0])
    e_chi=wrap_pi(chi_d-chi)
    phi_c=np.clip(1.8*e_chi,-max_bank,max_bank)
    h=-pos_ned[2]; h_d=-wp_ned[2]
    theta_c=np.clip(0.025*(h_d-h),np.deg2rad(-12),np.deg2rad(12))
    return phi_c,theta_c,chi_d

def update_autopilot(x,trim_u,Va,Va_c,phi_c,theta_c,state,dt,p):
    phi,theta=x[6],x[7]
    pb,qb,rb=x[9:12]
    p_c=4.0*(phi_c-phi)
    q_c=3.5*(theta_c-theta)
    ep=p_c-pb; eq=q_c-qb; ev=Va_c-Va
    state.int_p=np.clip(state.int_p+ep*dt,-1,1)
    state.int_q=np.clip(state.int_q+eq*dt,-1,1)
    state.int_v=np.clip(state.int_v+ev*dt,-10,10)
    da=0.20*ep+0.03*state.int_p-0.04*pb
    de=trim_u[1]-(0.23*eq+0.035*state.int_q-0.04*qb)
    dr=-0.18*rb
    throttle=trim_u[3]+0.035*ev+0.008*state.int_v
    target=np.array([da,de,dr])
    lim=np.array([p.aileron_limit_rad,p.elevator_limit_rad,p.rudder_limit_rad])
    target=np.clip(target,-lim,lim)
    delta=np.clip(target-state.prev_surfaces,-p.surface_rate_limit_rad_s*dt,p.surface_rate_limit_rad_s*dt)
    surf=state.prev_surfaces+delta
    state.prev_surfaces=surf.copy()
    return np.array([surf[0],surf[1],surf[2],np.clip(throttle,0,1)])
