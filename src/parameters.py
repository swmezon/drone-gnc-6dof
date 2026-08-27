from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuadrotorParams:
    """Physical parameters used by the rigid-body quadrotor model.

    The defaults returned by :func:`nominal_parameters` are intentionally
    labeled as nominal simulation values. They are not claimed to represent
    any specific commercial vehicle.
    """

    mass: float          # kg
    gravity: float       # m/s^2
    Ixx: float           # kg*m^2
    Iyy: float           # kg*m^2
    Izz: float           # kg*m^2

    @property
    def inertia(self) -> np.ndarray:
        """Return the 3x3 principal-axis inertia matrix."""
        return np.diag([self.Ixx, self.Iyy, self.Izz]).astype(float)

    def validate(self) -> None:
        """Raise ValueError if a physical parameter is non-positive."""
        values = {
            "mass": self.mass,
            "gravity": self.gravity,
            "Ixx": self.Ixx,
            "Iyy": self.Iyy,
            "Izz": self.Izz,
        }
        for name, value in values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive; got {value}.")


def nominal_parameters() -> QuadrotorParams:
    """Return a simple, self-consistent parameter set for simulation."""
    params = QuadrotorParams(
        mass=1.50,
        gravity=9.81,
        Ixx=0.020,
        Iyy=0.020,
        Izz=0.040,
    )
    params.validate()
    return params
