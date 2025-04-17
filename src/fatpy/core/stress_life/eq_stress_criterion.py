from abc import ABC, abstractmethod

import numpy as np

from fatpy.data_parsing.fe_model import FEModel
from fatpy.data_parsing.material import MaterialProperties
from fatpy.utilities.stress_correction import MeanStressCorrection


@abstractmethod
class EqStressCriterion(ABC):
    """Abstract base class for equivalent stress criteria."""

    @abstractmethod
    def calculate_eq_stress(
        self,
        fe_data: FEModel,
        material: MaterialProperties,
        eq_stress_correction: MeanStressCorrection,
    ) -> float:
        """Calculate equivalent stress based on the selected criterion.

        Args:
            fe_data: Finite Element model data containing stresses.
            material: Material properties.


        Returns:
            Equivalent stress value.
        """
        pass


class MansonMcKnight(EqStressCriterion):
    """Manson-McKnight equivalent stress criterion.

    This criterion is used for high-cycle fatigue analysis and is based on the
    equivalent alternating and mean stresses. It is defined as:

    $$ \\sigma_{eq} = \\sqrt{\\sigma_a^2 + \\sigma_m^2} $$

    where:
    - $\\sigma_{eq}$ is the equivalent stress
    - $\\sigma_a$ is the stress amplitude
    - $\\sigma_m$ is the mean stress
    """

    def calculate_eq_stress(
        self, fe_data: FEModel, material: MaterialProperties, correction: MeanStressCorrection
    ) -> float:
        """Calculate equivalent stress based on Manson-McKnight criterion.

        Args:
            fe_data: Finite Element model data containing stresses.
            material: Material properties.

        Returns:
            Equivalent stress value.
        """
        # Min and max stress values from FE data
        min_stress: float = fe_data.stress_tensor.min()
        max_stress: float = fe_data.stress_tensor.max()

        mean: float = (max_stress + min_stress) / 2
        amplitude: float = (max_stress - min_stress) / 2

        # dummy calculation
        sigma_m: float = np.sqrt(mean**2)
        sigma_a: float = np.sqrt(amplitude**2)

        return float(correction.correct_eq_stress_amplitude(sigma_a, sigma_m, material))
