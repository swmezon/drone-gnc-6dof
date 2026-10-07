import numpy as np

def R_body_to_ned(phi, theta, psi):
    cphi,sphi=np.cos(phi),np.sin(phi)
    cth,sth=np.cos(theta),np.sin(theta)
    cpsi,spsi=np.cos(psi),np.sin(psi)
    return np.array([
        [cth*cpsi, sphi*sth*cpsi-cphi*spsi, cphi*sth*cpsi+sphi*spsi],
        [cth*spsi, sphi*sth*spsi+cphi*cpsi, cphi*sth*spsi-sphi*cpsi],
        [-sth,     sphi*cth,                  cphi*cth],
    ])

def euler_rate_matrix(phi, theta):
    cphi,sphi=np.cos(phi),np.sin(phi)
    cth=np.cos(theta)
    tth=np.tan(theta)
    if abs(cth) < 1e-6:
        cth = np.sign(cth)*1e-6 if cth != 0 else 1e-6
    return np.array([
        [1.0, sphi*tth, cphi*tth],
        [0.0, cphi,     -sphi],
        [0.0, sphi/cth, cphi/cth],
    ])
