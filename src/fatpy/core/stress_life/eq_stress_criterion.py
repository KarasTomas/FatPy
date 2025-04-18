from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from fatpy.data_parsing.fe_model import FEModel
from fatpy.data_parsing.material import MaterialProperties
from fatpy.utilities.stress_correction import MeanStressCorrection


class EqStressCriterion(ABC):
    """Abstract base class for equivalent stress criteria."""

    @abstractmethod
    def calculate_eq_stress(
        self,
        fe_data: FEModel,
        material: MaterialProperties,
        eq_stress_correction: MeanStressCorrection,
    ) -> NDArray[np.float64]:
        """Calculate equivalent stress based on the selected criterion.
        This may include mean stress correction depending on the selected correction model.

        Args:
            fe_data: Finite Element model data containing stresses.
            material: Material properties.


        Returns:
            Equivalent stress field as a NumPy array.
        """
        pass


class MansonMcKnight(EqStressCriterion):
    """Manson-McKnight equivalent stress criterion.

    Computes equivalent stress for high-cycle fatigue using signed von Mises-based
    mean and amplitude stresses.

    $$
    \sigma_m = \operatorname{SIGN}(\sigma_{xm} + \sigma_{ym} + \sigma_{zm}) \cdot
    \frac{\sqrt{2}}{2} \cdot \sqrt{(\sigma_{xm} - \sigma_{ym})^2 + (\sigma_{ym} - \sigma_{zm})^2 + (\sigma_{zm} - \sigma_{xm})^2 + 6(\tau_{xym}^2 + \tau_{yzm}^2 + \tau_{zxm}^2)}
    $$

    $$
    \sigma_a = \frac{\sqrt{2}}{2} \cdot \sqrt{(\sigma_{xa} - \sigma_{ya})^2 + (\sigma_{ya} - \sigma_{za})^2 + (\sigma_{za} - \sigma_{xa})^2 + 6(\tau_{xya}^2 + \tau_{yza}^2 + \tau_{zxa}^2)}
    $$

    The equivalent stress is then corrected using the selected mean stress correction model.
    """

    def calculate_eq_stress(
        self, fe_data: FEModel, material: MaterialProperties, correction: MeanStressCorrection
    ) -> NDArray[np.float64]:
        """Calculate equivalent stress based on Manson-McKnight criterion.

        Args:
            fe_data: Finite Element model data containing stresses.
            material: Material properties.

        Returns:
            Equivalent stress value.
        """
        # Min and max stress values from FE data
        min_stress: NDArray[np.float64] = fe_data.stress_tensor.min()
        max_stress: NDArray[np.float64] = fe_data.stress_tensor.max()

        mean: NDArray[np.float64] = (max_stress + min_stress) / 2
        amplitude: NDArray[np.float64] = (max_stress - min_stress) / 2

        # dummy calculation
        sigma_m: NDArray[np.float64] = np.sqrt(mean**2)
        sigma_a: NDArray[np.float64] = np.sqrt(amplitude**2)

        return correction.eq_stress_amplitude(sigma_a, sigma_m, material)
