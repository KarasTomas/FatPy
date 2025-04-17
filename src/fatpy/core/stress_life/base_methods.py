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

        """
        pass


class StressInvariant(StressLifeMethod):
    """Stress invariant method for fatigue analysis.

    This method calculates the equivalent stress using the stress invariants.
    It is suitable for complex loading conditions and provides a more accurate
    representation of the stress state.
    """

    def eq_stress(
        self,
        material: MaterialProperties,
        fe_model: FEModel,
        eq_stress_criterion: EqStressCriterion,
        eq_stress_correction: MeanStressCorrection,
        load_case: Optional[LoadCase] = None,
    ) -> None:
        """Calculate equivalent stress based on stress invariants.

        Args:
            material: Material properties.
            fe_model: Finite Element model data containing stresses.
            eq_stress_criterion: Selected equivalent stress criterion.
            eq_stress_correction: Mean stress correction method.
            load_case: Optional load case data (e.g., for scaling FE stresses).

        Returns:

        """
        eq_stress: NDArray[np.float64] = eq_stress_criterion.calculate_eq_stress(
            fe_model, material, eq_stress_correction
        )
        fe_model.add_eq_stress(eq_stress)
        return
