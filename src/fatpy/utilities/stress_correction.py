from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from fatpy.data_parsing.material import MaterialProperties


class MeanStressCorrection(ABC):
    """Abstract base class for mean stress correction methods."""

    @abstractmethod
    def eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Calculates equivalent stress amplitude based on selected mean stress correction method.

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

    Applies a linear correction using the ultimate tensile strength (UTS).
    Conservative for tensile mean stresses; commonly used for brittle materials.

    $$ \sigma_{a,eq} = \frac{\sigma_a}{1 - \frac{\sigma_m}{\sigma_{UTS}}} $$

    where:
    - $\sigma_{a,eq}$ is the equivalent stress amplitude
    - $\sigma_a$ is the stress amplitude
    - $\sigma_m$ is the mean stress
    - $\sigma_{UTS}$ is the ultimate tensile strength
    """

    def eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Calculates equivalent stress amplitude based on Goodman mean stress correction.

        Args:
            stress_amplitude: Stress amplitude
            mean_stress: Mean stress
            material: Material properties

        Returns:
            Corrected equivalent stress amplitude value.

        """
        UTS = material.ultimate_tensile_strength
        stress_amplitude = np.asarray(stress_amplitude)
        mean_stress = np.asarray(mean_stress)
        eq_stress = np.where(
            mean_stress <= 0,
            stress_amplitude,
            stress_amplitude / (1 - mean_stress / UTS),
        )
        return eq_stress


class GerberCorrection(MeanStressCorrection):
    """Gerber mean stress correction method.

    Uses a parabolic relation with UTS. More accurate for ductile materials,
    but not valid for high compressive mean stresses.

    $$ \sigma_{a,eq} = \frac{\sigma_a}{1 - (\frac{\sigma_m}{\sigma_{UTS}})^2} $$
    """

    def eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Calculates equivalent stress amplitude based on Gerber mean stress correction.

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

    Computes a fatigue damage parameter as the product of max stress and amplitude.
    Suitable for low-cycle fatigue and high mean stress.

    $$ \sigma_{a,eq} = \sqrt{\sigma_{max} \cdot \sigma_a} = \sqrt{(\sigma_m + \sigma_a) \cdot \sigma_a} $$
    """

    def eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Calculates equivalent stress amplitude based on SWT mean stress correction.

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

    Linear correction using the fatigue strength coefficient $\sigma_f'$.
    Useful for strain-life models and low-cycle fatigue.

    $$ \sigma_{a,eq} = \frac{\sigma_a}{1 - \frac{\sigma_m}{\sigma_f'}} $$

    """

    def eq_stress_amplitude(
        self, stress_amplitude: NDArray[np.float64], mean_stress: NDArray[np.float64], material: MaterialProperties
    ) -> NDArray[np.float64]:
        """Calculates equivalent stress amplitude based on Morrow mean stress correction.

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
