"""Stress invariant criterions for fatigue life prediction."""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def manson_mcknight_criterion[T: np.floating](
    min_stress_tensor: NDArray[T], max_stress_tensor: NDArray[T]
) -> Tuple[NDArray[T], NDArray[T]]:
    """Calculate Manson-McKnight equivalent mean stress and amplitude.

    Args:
        min_stress_tensor: Minimum stress tensor, shape (n, 6) for n evaluation points.
            Components: [Sxx, Syy, Szz, Sxy, Syz, Szx] #! Naming convention?
        max_stress_tensor: Maximum stress tensor, shape (n, 6).
            Components: [Sxx, Syy, Szz, Sxy, Syz, Szx]

    Returns:
        Tuple of (eq_mean_stress, eq_amplitude_stress), each with shape (n,) containing
        scalar equivalent stresses for each evaluation point.
    """
    mean_stress = (min_stress_tensor + max_stress_tensor) / 2
    stress_amp = (max_stress_tensor - min_stress_tensor) / 2

    sxx_mean, syy_mean, szz_mean, sxy_mean, syz_mean, szx_mean = [
        mean_stress[:, i] for i in range(6)
    ]
    sxx_amp, syy_amp, szz_amp, sxy_amp, syz_amp, szx_amp = [
        stress_amp[:, i] for i in range(6)
    ]

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
    eq_stress_amp = vm_amp

    return (eq_mean_stress, eq_stress_amp)
