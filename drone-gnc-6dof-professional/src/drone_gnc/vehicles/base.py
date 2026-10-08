from abc import ABC, abstractmethod
import numpy as np

class VehicleModel(ABC):
    """Common interface for any vehicle dynamics model used by the GNC stack."""
    state_size: int
    control_size: int

    @abstractmethod
    def derivatives(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Return x_dot = f(x,u)."""
        raise NotImplementedError

    @abstractmethod
    def nominal_control(self) -> np.ndarray:
        """Return a nominal operating/equilibrium control input when applicable."""
        raise NotImplementedError
