from abc import ABC, abstractmethod
from typing import Optional

from fatpy.data_parsing.fe_model import FEModel
from fatpy.data_parsing.loads import LoadCase
from fatpy.data_parsing.material import MaterialProperties


class StressLifeMethod(ABC):
    """Abstract base class for stress-life fatigue analysis methods."""

    @abstractmethod
    def eq_stress(self, material: MaterialProperties, fe_model: FEModel, load_case: Optional[LoadCase] = None) -> float:
        """Calculate equivalent stress based on selected Criterion and CorrectionMethod.

        Args:
            material: Material properties.
            fe_model: Finite Element model data containing stresses.
            load_case: Optional load case data (e.g., for scaling FE stresses).

        Returns:
            Equivalent stress value.
        """
        pass
