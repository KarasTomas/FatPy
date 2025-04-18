from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray

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
    ) -> None:
        """Calculate equivalent stress based on selected Criterion and CorrectionMethod.

        Args:
            material: Material properties.
            fe_model: Finite Element model data containing stresses.
            eq_stress_criterion: Selected equivalent stress criterion.
            eq_stress_correction: Mean stress correction method.
            load_case: Optional load case data (e.g., for scaling FE stresses).

        Returns:
            None: The method modifies the FEModel in place by adding the equivalent stress.
        """
        pass


class StressInvariant(StressLifeMethod):
    """Stress invariant method for fatigue analysis."""

    def eq_stress(
        self,
        material: MaterialProperties,
        fe_model: FEModel,
        eq_stress_criterion: EqStressCriterion,
        eq_stress_correction: MeanStressCorrection,
        load_case: Optional[LoadCase] = None,
    ) -> None:
        """Calculate equivalent stress based on selected Criterion and CorrectionMethod.

        Args:
            material: Material properties.
            fe_model: Finite Element model data containing stresses.
            eq_stress_criterion: Selected equivalent stress criterion.
            eq_stress_correction: Mean stress correction method.
            load_case: Optional load case data (e.g., for scaling FE stresses).

        Returns:
            None: The method modifies the FEModel in place by adding the equivalent stress.
        """
        eq_stress: NDArray[np.float64] = eq_stress_criterion.calculate_eq_stress(
            fe_model, material, eq_stress_correction
        )
        fe_model.add_stress_column(eq_stress)
        return
