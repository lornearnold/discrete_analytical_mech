"""Sample class for representing particle samples."""

import warnings

import numpy as np

from .base import IndexedSet
from .utils import sphere_vol


class Sample(IndexedSet):
    """
    A class representing a sample with particle sizes and associated data.
    Inherits from IndexedSet to maintain sorted sizes and corresponding data arrays.
    """

    def __init__(
        self,
        sizes,
        quantities,
        shape="sphere",
        density=1.0,
    ):
        """
        Initialize Sample with particle sizes and associated data.

        Parameters:
        -----------
        sizes : array-like
            Array of particle size values that must be unique and positive
        quantities : array-like
            Array of particle quantity values that must be positive integers.
        shape : str, optional
            Shape of particles (default: "sphere")
        density : float, optional
            Density of particles (default: 1.0)
        """
        # Call parent constructor with just sizes
        super().__init__(sizes)

        # Validate that sizes are positive (specific to Sample)
        if np.any(self.sizes <= 0):
            raise ValueError("All particle sizes must be positive")

        self.shape = shape
        self.density = density

        # Sort quantities to match sorted sizes
        quantities = self._sort_like_sizes(quantities)

        # Validate quantities
        quantities = np.asarray(quantities)
        if np.any(quantities < 0):
            raise ValueError("All quantities must be positive integers")

        # Handle zero quantities
        if np.any(quantities == 0):
            warnings.warn("Zero quantities found. Removing corresponding entries.")
            non_zero_mask = quantities > 0
            quantities = self._filter_by_mask(non_zero_mask, quantities)

        # Handle floating-point quantities
        if not all(isinstance(q, (int, np.integer)) for q in quantities):
            warnings.warn("Floating-point quantities found. Converting to integers.")
            quantities = np.round(quantities).astype(int)

        self.quantities = quantities
        self.total_masses = (
            np.asarray(self.quantities) * sphere_vol(self.sizes) * self.density
        )

    def norm_1(self):
        """
        Return the L-1 norm of the sample quantities, which is the total number of particles.
        """
        return np.sum(self.quantities)
