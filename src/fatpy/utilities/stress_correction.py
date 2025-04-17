from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from fatpy.data_parsing.material import MaterialProperties


class MeanStressCorrection(ABC):
    """Abstract base class for mean stress correction methods."""

    @abstractmethod
    def correct_eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Correct stress amplitude based on selected method.

        Args:
            stress_amplitude: Stress amplitude
            mean_stress: Mean stress
            material: Material properties

        Returns:
            Corrected equivalent stress amplitude value.
        """
        pass


class GoodmanCorrection(MeanStressCorrection):
    """Goodman mean stress correction method.

    The Goodman relation provides a linear damage rule connecting the fatigue limit
    for a specified number of cycles and the ultimate tensile strength.

    $$ \\sigma_{a,eq} = \\frac{\\sigma_a}{1 - \\frac{\\sigma_m}{\\sigma_{UTS}}} $$

    where:
    - $\\sigma_{a,eq}$ is the equivalent stress amplitude
    - $\\sigma_a$ is the stress amplitude
    - $\\sigma_m$ is the mean stress
    - $\\sigma_{UTS}$ is the ultimate tensile strength
    """

    def correct_eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Correct stress amplitude based on Goodman mean stress correction.

        Args:
            stress_amplitude: Stress amplitude
            mean_stress: Mean stress
            material: Material properties

        Returns:
            Corrected equivalent stress amplitude value.

        """

        if mean_stress <= 0:
            return stress_amplitude

        return stress_amplitude / (1 - mean_stress / material.ultimate_tensile_strength)


class GerberCorrection(MeanStressCorrection):
    """Gerber mean stress correction method.

    The Gerber relation provides a quadratic damage rule connecting the fatigue limit
    for a specified number of cycles and the ultimate tensile strength.

    $$ \\sigma_{a,eq} = \\frac{\\sigma_a}{1 - (\\frac{\\sigma_m}{\\sigma_{UTS}})^2} $$
    """

    def correct_eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Correct stress amplitude based on Gerber mean stress correction.

        Args:
            stress_amplitude: Stress amplitude
            mean_stress: Mean stress
            material: Material properties

        Returns:
            Corrected equivalent stress amplitude value.

        """

        if mean_stress <= 0:
            return stress_amplitude

        return stress_amplitude / (1 - (mean_stress / material.ultimate_tensile_strength) ** 2)


class SWTCorrection(MeanStressCorrection):
    """Smith-Watson-Topper mean stress correction method.

    The SWT parameter is defined as the product of the maximum stress and
    the stress amplitude.

    $$ \\sigma_{a,eq} = \\sqrt{\\sigma_{max} \\cdot \\sigma_a} = \\sqrt{(\\sigma_m + \\sigma_a) \\cdot \\sigma_a} $$
    """

    def correct_eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Correct stress amplitude based on SWT mean stress correction.

        Args:
            stress_amplitude: Stress amplitude
            mean_stress: Mean stress
            material: Material properties

        Returns:
            Corrected equivalent stress amplitude value.
        """
        max_stress = mean_stress + stress_amplitude

        if max_stress <= 0:
            return stress_amplitude

        return np.sqrt(max_stress * stress_amplitude)


class MorrowCorrection(MeanStressCorrection):
    """Morrow mean stress correction method.

    The Morrow relation is similar to Goodman but uses the true fracture strength
    instead of the ultimate tensile strength.

    $$ \\sigma_{a,eq} = \\frac{\\sigma_a}{1 - \\frac{\\sigma_m}{\\sigma_f'}} $$

    where $\\sigma_f'$ is the fatigue strength coefficient.
    """

    def correct_eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Correct stress amplitude based on Morrow mean stress correction.

        Args:
            stress_amplitude: Stress amplitude
            mean_stress: Mean stress
            material: Material properties

        Returns:
            Corrected equivalent stress amplitude value.

        """
        if mean_stress <= 0:
            return stress_amplitude

        return stress_amplitude / (1 - mean_stress / material.fatigue_strength_coefficient)
