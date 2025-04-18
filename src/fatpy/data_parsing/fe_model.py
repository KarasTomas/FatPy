from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class FEModel:
    """
    Represents data extracted from a Finite Element model.

    Intended structure conceptually similar to a DataFrame with MultiIndex
    (element_id, node_id) and columns for variables like stress, strain, etc.
    Pylife mesh structure.
    """

    element_id: NDArray[np.int_]  # Shape (num_elements,)
    node_id: NDArray[np.int_]  # Shape (num_nodes,)
    # Shape (num_nodes, 3) -> columns X, Y, Z
    node_coordinates: NDArray[np.float64]
    # Shape (num_stress_points, 6) -> columns S11, S22, S33, S12, S23, S13
    stress_tensor: NDArray[np.float64]
    # Shape (num_elements, 1) -> Keys are elements, value is list of node_ids
    connectivity: Dict[int, List[int]]

    # Computed map: node_id -> list of element_ids connected to that node
    node_element_map: Dict[int, List[int]] = field(init=False)

    def __post_init__(self) -> None:
        """Validate shapes and compute node-element map."""

        # if self.node_coordinates.ndim != 2 or self.node_coordinates.shape[1] != 3:
        #     raise ValueError(f"node_coordinates must have shape (num_nodes, 3), but got {self.node_coordinates.shape}")

        # if self.stress_tensor.ndim != 2 or self.stress_tensor.shape[1] != 6:
        #     raise ValueError(
        #         f"stress_tensor must have shape (num_stress_points, 6), but got {self.stress_tensor.shape}"
        #     )

        # provisional --- Compute Flipped Connectivity (Node -> Elements Map) ---
        self.node_element_map = {}
        # Iterate through the connectivity dictionary (element_id: [node_id_list])
        for element_id, node_list in self.connectivity.items():
            for node_id in node_list:
                if node_id not in self.node_element_map:
                    self.node_element_map[node_id] = []
                self.node_element_map[node_id].append(element_id)

    def add_stress_column(self, stress_col: NDArray[np.float64]) -> None:
        """Add stress column to FE data.

        Args:
            stress_col: Column of stress values.
        """
        self.stress_tensor = np.column_stack((self.stress_tensor, stress_col))
