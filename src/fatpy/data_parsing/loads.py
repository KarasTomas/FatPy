from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class LoadCase:
    """Load case for stress life analysis."""

    name: str
    stress_amplitude: Union[float, NDArray[np.float64]]
    mean_stress: Union[float, NDArray[np.float64]]
    cycles: Union[float, NDArray[np.float64]]
