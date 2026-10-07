from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AircraftParams:
    mass: float = 13.5
    rho: float = 1.225
    S: float = 0.55
    b: float = 2.90
    c: float = 0.19
    g: float = 9.80665
    Jx: float = 0.8244
    Jy: float = 1.135
    Jz: float = 1.759
    Jxz: float = 0.1204
    thrust_max: float = 55.0

    CL0: float = 0.28
    CL_alpha: float = 3.45
    CL_q: float = 0.0
    CL_de: float = -0.36
    CD0: float = 0.035
    CD_alpha2: float = 0.30
    Cm0: float = -0.023
    Cm_alpha: float = -0.38
    Cm_q: float = -3.6
    Cm_de: float = -0.50

    CY_beta: float = -0.98
    CY_p: float = -0.26
    CY_r: float = 0.14
    CY_da: float = 0.08
    CY_dr: float = -0.17
    Cl_beta: float = -0.12
    Cl_p: float = -0.26
    Cl_r: float = 0.14
    Cl_da: float = 0.08
    Cl_dr: float = 0.105
    Cn_beta: float = 0.25
    Cn_p: float = 0.022
    Cn_r: float = -0.35
    Cn_da: float = 0.06
    Cn_dr: float = -0.032

    aileron_limit_rad: float = np.deg2rad(25.0)
    elevator_limit_rad: float = np.deg2rad(25.0)
    rudder_limit_rad: float = np.deg2rad(30.0)
    surface_rate_limit_rad_s: float = np.deg2rad(80.0)


def nominal_aircraft() -> AircraftParams:
    return AircraftParams()
