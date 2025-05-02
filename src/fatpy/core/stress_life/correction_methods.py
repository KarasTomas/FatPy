"""Correction methods for the stress-life."""

import numpy as np
from numpy.typing import NDArray

# ? How to handle different methods requiring different number parameters - consistency?


def calc_goodman_eq_stress[T: np.floating](
    stress_amp: NDArray[T],
    mean_stress: NDArray[T],
    ultimate_tensile_strength: float,
) -> NDArray[T]:
    """Calculate Goodman equivalent stress.

    Args:
    stress_amp (T): Stress amplitude vector of shape (n, 6), n is the number of points.
    mean_stress (T): Mean stress vector of shape (n, 6), n is the number of points.
    ultimate_tensile_strength (float): Ultimate tensile strength.

    Returns:
    eq_stress (T): Goodman equivalent stress correction.
    Vector of shape (n, 1), n is the number of points.
    """
    eq_stress = np.where(
        mean_stress <= 0,
        stress_amp,
        stress_amp / (1 - mean_stress / ultimate_tensile_strength),
    )
    return eq_stress


def calc_gerber_eq_stress[T: np.floating](
    stress_amp: NDArray[T],
    mean_stress: NDArray[T],
    ultimate_tensile_strength: float,
) -> NDArray[T]:
    """Calculate Gerber equivalent stress.

    Args:
    stress_amp (T): Stress amplitude vector of shape (n, 6), n is the number of points.
    mean_stress (T): Mean stress vector of shape (n, 6), n is the number of points.
    ultimate_tensile_strength (float): Ultimate tensile strength.

    Returns:
    eq_stress (T): Gerber equivalent stress correction.
    Vector of shape (n, 1), n is the number of points.
    """
    eq_stress = np.where(
        mean_stress <= 0,
        stress_amp,
        stress_amp / (1 - (mean_stress / ultimate_tensile_strength) ** 2),
    )
    return eq_stress


def calc_swt_eq_stress[T: np.floating](
    stress_amp: NDArray[T],
    mean_stress: NDArray[T],
    *args: float,  # Added for consistency
) -> NDArray[T]:
    """Calculate Smith-Watson-Topper (SWT) equivalent stress.

    Args:
    stress_amp (T): Stress amplitude vector of shape (n, 6), n is the number of points.
    mean_stress (T): Mean stress vector of shape (n, 6), n is the number of points.
    *args: Not used by SWT, included for consistent interface with other methods.

    Returns:
    eq_stress (T): SWT equivalent stress correction.
    Vector of shape (n, 1), n is the number of points.
    """
    max_stress = mean_stress + stress_amp
    eq_stress = np.where(
        max_stress <= 0,
        stress_amp,
        np.sqrt(max_stress * stress_amp),
    )
    return eq_stress
