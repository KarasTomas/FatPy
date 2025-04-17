from dataclasses import dataclass
from typing import Optional


@dataclass
class MaterialProperties:
    """Material properties required for stress life analysis."""

    name: str
    ultimate_tensile_strength: float
    yield_strength: float
    elastic_modulus: float
    poissons_ratio: float
    sheer_modulus: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate material properties."""
        if self.ultimate_tensile_strength <= 0:
            raise ValueError("Ultimate tensile strength must be positive")
        if self.yield_strength <= 0:
            raise ValueError("Yield strength must be positive")
        if self.elastic_modulus <= 0:
            raise ValueError("Elastic modulus must be positive")
        if self.poissons_ratio <= 0:
            raise ValueError("Poisson's ratio must be positive")
