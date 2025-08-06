"""Base IndexedSet class for gsd_lib package."""

import numpy as np


class IndexedSet:
    """
    Base class for indexed sets with sizes.
    Ensures sizes are unique and sorted, and provides utilities for child classes
    to maintain attributes that should be indexed with sizes.
    """

    def __init__(self, sizes):
        """
        Initialize IndexedSet with sizes array.

        Parameters:
        -----------
        sizes : array-like
            Array of size values that must be unique and will be sorted
        """
        # Convert to numpy array
        sizes = np.asarray(sizes)

        # Check for unique sizes
        if len(np.unique(sizes)) != len(sizes):
            print(sizes)
            raise ValueError("All entries in sizes must be unique")

        # Get sorting indices and store them for child classes
        self._sort_indices = np.argsort(sizes)

        # Sort sizes
        self.sizes = sizes[self._sort_indices]

    def _sort_like_sizes(self, *arrays):
        """
        Sort one or more arrays using the same indices that were used to sort sizes.

        Parameters:
        -----------
        *arrays : array-like
            Arrays to be sorted in the same order as sizes

        Returns:
        --------
        tuple or single array
            Sorted arrays in the same order as sizes
        """
        sorted_arrays = []
        for arr in arrays:
            arr = np.asarray(arr)
            if len(arr) != len(self.sizes):
                raise ValueError(
                    f"Array length {len(arr)} must match sizes length {len(self.sizes)}"
                )
            sorted_arrays.append(arr[self._sort_indices])

        return sorted_arrays[0] if len(sorted_arrays) == 1 else tuple(sorted_arrays)

    def _filter_by_mask(self, mask, *arrays):
        """
        Filter sizes and associated arrays using a boolean mask.
        Updates self.sizes and returns filtered arrays.

        Parameters:
        -----------
        mask : array-like of bool
            Boolean mask to apply
        *arrays : array-like
            Arrays to be filtered along with sizes

        Returns:
        --------
        tuple or single array
            Filtered arrays
        """
        self.sizes = self.sizes[mask]

        filtered_arrays = []
        for arr in arrays:
            arr = np.asarray(arr)
            filtered_arrays.append(arr[mask])

        return (
            filtered_arrays[0] if len(filtered_arrays) == 1 else tuple(filtered_arrays)
        )

    def __len__(self):
        """
        Return the number of unique sizes in the set.
        """
        return len(self.sizes)
