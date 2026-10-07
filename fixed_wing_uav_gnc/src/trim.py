import numpy as np
from scipy.optimize import least_squares
from src.dynamics import derivative

def trim_state_control(p, Va_target=25.0, altitude_m=100.0):
    def unpack(z):
        alpha,theta,de,throttle=z
        x=np.zeros(12)
        x[2]=-altitude_m
        x[3]=Va_target*np.cos(alpha)
        x[5]=Va_target*np.sin(alpha)
        x[7]=theta
        u=np.array([0.0,de,0.0,throttle])
        return x,u
    def residual(z):
        x,u=unpack(z)
        dx=derivative(0.0,x,u,p)
        return np.array([dx[3],dx[5],dx[10], z[1]-z[0]])
    sol=least_squares(residual, x0=np.array([0.05,0.05,-0.05,0.55]), bounds=([-0.2,-0.2,-0.45,0.05],[0.25,0.25,0.45,1.0]))
    x,u=unpack(sol.x)
    return x,u,sol
