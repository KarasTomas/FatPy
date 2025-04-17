from abc import ABC, abstractmethod
from typing import Optional

from fatpy.core.stress_life.eq_stress_criterion import EqStressCriterion
from fatpy.data_parsing.fe_model import FEModel
from fatpy.data_parsing.loads import LoadCase
from fatpy.data_parsing.material import MaterialProperties
from fatpy.utilities.stress_correction import MeanStressCorrection


class StressLifeMethod(ABC):
    """Abstract base class for stress-life fatigue analysis methods."""

    @abstractmethod
    def eq_stress(
        self,
        material: MaterialProperties,
        fe_model: FEModel,
        eq_stress_criterion: EqStressCriterion,
        eq_stress_correction: MeanStressCorrection,
        load_case: Optional[LoadCase] = None,
    ) -> float:
        """Calculate equivalent stress based on selected Criterion and CorrectionMethod.

        Args:
            material: Material properties.
            fe_model: Finite Element model data containing stresses.
            eq_stress_criterion: Selected equivalent stress criterion.
            eq_stress_correction: Mean stress correction method.
            load_case: Optional load case data (e.g., for scaling FE stresses).

        Returns:
            Equivalent stress value.
        """
        pass
