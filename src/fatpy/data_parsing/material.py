"""Material properties parsing module."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaterialProperties:
    """Material properties required for fatigue life analysis."""

    name: str
    ultimate_tensile_strength: float
    yield_strength: float
    elastic_modulus: float
    poisson_ratio: float
    fatigue_strength_coefficient: Optional[float] = None
    shear_modulus: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate material properties."""
        if self.ultimate_tensile_strength <= 0:
            raise ValueError("Ultimate tensile strength must be positive")
        if self.yield_strength <= 0:
            raise ValueError("Yield strength must be positive")
        if self.elastic_modulus <= 0:
            raise ValueError("Elastic modulus must be positive")
        if self.poisson_ratio <= 0:
            raise ValueError("Poisson's ratio must be positive")

    def calc_shear_modulus(self) -> float:
        """Calculate shear modulus based on elastic modulus and Poisson's ratio."""
        if self.shear_modulus is None:
            self.shear_modulus = self.elastic_modulus / (2 * (1 + self.poisson_ratio))
        return self.shear_modulus
