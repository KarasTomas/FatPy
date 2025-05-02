from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from fatpy.data_parsing.fe_model import FEModel
from fatpy.data_parsing.material import MaterialProperties
from fatpy.utilities.stress_correction import MeanStressCorrection

# TODO: Stress tensor vs FE data, tensor definition?
NDArray2D = np.ndarray[tuple[int, int], np.dtype[np.floating]]  # 2D array


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


def manson_mcknight_criterion[T: np.floating](
    min_stress_tensor: NDArray[T], max_stress_tensor: NDArray[T]
) -> Tuple[NDArray[T], NDArray[T]]:
    """Calculate Manson-McKnight equivalent mean stress and amplitude.

    Args:
        min_stress_tensor: Minimum stress tensor, shape (n, 6) for n evaluation points.
            Components: [Sxx, Syy, Szz, Sxy, Syz, Szx]
        max_stress_tensor: Maximum stress tensor, shape (n, 6).
            Components: [Sxx, Syy, Szz, Sxy, Syz, Szx]

    Returns:
        Tuple of (eq_mean_stress, eq_amplitude_stress), each with shape (n,) containing
        scalar equivalent stresses for each evaluation point.
    """

    mean_stress_tensor = (min_stress_tensor + max_stress_tensor) / 2
    amplitude_stress_tensor = (max_stress_tensor - min_stress_tensor) / 2

    sxx_mean, syy_mean, szz_mean, sxy_mean, syz_mean, szx_mean = [mean_stress_tensor[:, i] for i in range(6)]
    sxx_amp, syy_amp, szz_amp, sxy_amp, syz_amp, szx_amp = [amplitude_stress_tensor[:, i] for i in range(6)]

    hydrostatic_mean = sxx_mean + syy_mean + szz_mean

    vm_mean = 0.5 * np.sqrt(
        (sxx_mean - syy_mean) ** 2
        + (syy_mean - szz_mean) ** 2
        + (szz_mean - sxx_mean) ** 2
        + 6 * (sxy_mean**2 + syz_mean**2 + szx_mean**2)
    )

    vm_amp = 0.5 * np.sqrt(
        (sxx_amp - syy_amp) ** 2
        + (syy_amp - szz_amp) ** 2
        + (szz_amp - sxx_amp) ** 2
        + 6 * (sxy_amp**2 + syz_amp**2 + szx_amp**2)
    )

    eq_mean_stress = np.sign(hydrostatic_mean) * vm_mean
    eq_amplitude_stress = vm_amp

    return (eq_mean_stress, eq_amplitude_stress)


def dang_van_criterion[T: np.floating](
    stress_tensor: NDArray[T],
    material_parameters: Tuple[float, float],  # (alpha, tau_limit)
) -> NDArray[T]:
    """Calculate equivalent stress based on Dang Van criterion.

    The Dang Van criterion is a critical plane multiaxial fatigue criterion that considers
    the combined effect of microscopic shear stress and hydrostatic pressure. It states that
    fatigue damage occurs when:

    τ_max(t) + α * p(t) > τ_limit

    where:
    - τ_max is the maximum microscopic shear stress
    - p is the hydrostatic stress
    - α is a material parameter
    - τ_limit is the fatigue limit in pure shear

    Args:
        stress_tensor: Stress tensor history, shape (n, 6) for n evaluation points.
            Components: [Sxx, Syy, Szz, Sxy, Syz, Szx]
        material_parameters: Tuple containing (alpha, tau_limit)
            - alpha: Material parameter relating hydrostatic stress sensitivity
            - tau_limit: Fatigue limit in pure shear

    Returns:
        Equivalent stress with shape (n,) containing scalar values for each evaluation point.
    """
    alpha, tau_limit = material_parameters

    sxx, syy, szz, sxy, syz, szx = [stress_tensor[:, i] for i in range(6)]

    # Calculate hydrostatic pressure (negative of mean normal stress)
    hydrostatic_pressure = -(sxx + syy + szz) / 3

    # Calculate deviatoric stress components
    s_xx = sxx + hydrostatic_pressure
    s_yy = syy + hydrostatic_pressure
    s_zz = szz + hydrostatic_pressure
    s_xy = sxy
    s_yz = syz
    s_zx = szx

    # Calculate the eigenvalues of the deviatoric stress tensor
    J2 = 0.5 * (s_xx**2 + s_yy**2 + s_zz**2) + s_xy**2 + s_yz**2 + s_zx**2

    # Maximum microscopic shear stress (approximation using J2)
    tau_max = np.sqrt(J2)

    # Calculate Dang Van equivalent stress
    eq_stress = tau_max + alpha * hydrostatic_pressure

    # The criterion states eq_stress < tau_limit for safety
    # Return eq_stress for comparison with tau_limit
    return eq_stress
