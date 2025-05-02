"""Finite element model parsing module."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class FEModel:
    """Represents data extracted from a Finite Element model.

    Intended data structure is a vector of (n,variables).
    Where n is the number of elements or nodes.
    Number of variables corresponds to coordinates (3 columns), stress (6 columns), etc.

    Args:
        element_id: Array of element IDs.
        node_id: Array of node IDs.
        node_coordinates: Array of node coordinates, shape (n, 3).
        stress_tensor: Array of stress tensor values, shape (n, 6).
        connectivity: Dictionary mapping element IDs to lists of node IDs.
    """

    element_id: NDArray[np.uint32]
    node_id: NDArray[np.uint32]
    node_coordinates: NDArray[np.floating]
    stress_tensor: NDArray[np.floating]

    # TODO Revise type hint!
    connectivity: Dict[int, List[int]]
    # Computed map: node_id -> list of element_ids connected to that node
    node_element_map = None

    def add_stress_column[T: NDArray[np.floating]](self, stress_col: T) -> None:
        """Add stress column to FE data.

        Args:
            stress_col: Column of stress values.
        """
        self.stress_tensor = np.column_stack((self.stress_tensor, stress_col))
