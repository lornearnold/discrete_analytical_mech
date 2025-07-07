"""Tests for IndexedSet class."""

import numpy as np
import pytest

from gsd_lib import IndexedSet


class TestIndexedSet:
    """Test cases for IndexedSet class."""

    def test_basic_initialization(self):
        """Test basic initialization with valid inputs."""
        sizes = [1, 2, 3]
        indexed_set = IndexedSet(sizes)

        np.testing.assert_array_equal(indexed_set.sizes, [1, 2, 3])
        assert len(indexed_set) == 3

    def test_unsorted_sizes_get_sorted(self):
        """Test that unsorted sizes are automatically sorted."""
        sizes = [3, 1, 2]
        indexed_set = IndexedSet(sizes)

        # Sizes should be sorted
        np.testing.assert_array_equal(indexed_set.sizes, [1, 2, 3])
        # Check that sort indices are stored correctly
        assert hasattr(indexed_set, "_sort_indices")
        np.testing.assert_array_equal(indexed_set._sort_indices, [1, 2, 0])

    def test_numpy_array_inputs(self):
        """Test initialization with numpy arrays."""
        sizes = np.array([5.5, 2.2, 3.3])
        indexed_set = IndexedSet(sizes)

        np.testing.assert_array_equal(indexed_set.sizes, [2.2, 3.3, 5.5])
        assert isinstance(indexed_set.sizes, np.ndarray)

    def test_duplicate_sizes_raises_error(self):
        """Test that duplicate sizes raise ValueError."""
        sizes = [1, 2, 2, 3]

        with pytest.raises(ValueError, match="All entries in sizes must be unique"):
            IndexedSet(sizes)

    def test_empty_arrays(self):
        """Test initialization with empty arrays."""
        sizes = []
        indexed_set = IndexedSet(sizes)

        assert len(indexed_set.sizes) == 0
        assert len(indexed_set) == 0

    def test_single_element(self):
        """Test initialization with single element arrays."""
        sizes = [5.0]
        indexed_set = IndexedSet(sizes)

        np.testing.assert_array_equal(indexed_set.sizes, [5.0])
        assert len(indexed_set) == 1

    def test_negative_sizes(self):
        """Test that negative sizes are handled (should be sorted correctly)."""
        sizes = [-1, 3, 0, 2]
        indexed_set = IndexedSet(sizes)

        np.testing.assert_array_equal(indexed_set.sizes, [-1, 0, 2, 3])

    def test_float_sizes(self):
        """Test with floating point sizes."""
        sizes = [1.1, 2.2, 1.5]
        indexed_set = IndexedSet(sizes)

        np.testing.assert_array_equal(indexed_set.sizes, [1.1, 1.5, 2.2])

    def test_large_arrays(self):
        """Test with larger arrays."""
        n = 1000
        sizes = np.random.permutation(range(n))
        indexed_set = IndexedSet(sizes)

        # Check that sizes are sorted
        assert np.all(indexed_set.sizes[:-1] <= indexed_set.sizes[1:])
        # Check that we have the same number of elements
        assert len(indexed_set.sizes) == n
        assert len(indexed_set) == n

    def test_sort_like_sizes_single_array(self):
        """Test _sort_like_sizes method with single array."""
        sizes = [3, 1, 2]
        quantities = [30, 10, 20]
        indexed_set = IndexedSet(sizes)

        sorted_quantities = indexed_set._sort_like_sizes(quantities)

        # Should return single array, not tuple
        assert isinstance(sorted_quantities, np.ndarray)
        np.testing.assert_array_equal(sorted_quantities, [10, 20, 30])

    def test_sort_like_sizes_multiple_arrays(self):
        """Test _sort_like_sizes method with multiple arrays."""
        sizes = [3, 1, 2]
        quantities = [30, 10, 20]
        weights = [0.6, 0.2, 0.4]
        indexed_set = IndexedSet(sizes)

        sorted_quantities, sorted_weights = indexed_set._sort_like_sizes(
            quantities, weights
        )

        np.testing.assert_array_equal(sorted_quantities, [10, 20, 30])
        np.testing.assert_array_equal(sorted_weights, [0.2, 0.4, 0.6])

    def test_sort_like_sizes_length_mismatch(self):
        """Test that _sort_like_sizes raises error for mismatched lengths."""
        sizes = [1, 2, 3]
        quantities = [10, 20]  # Wrong length
        indexed_set = IndexedSet(sizes)

        with pytest.raises(
            ValueError, match="Array length 2 must match sizes length 3"
        ):
            indexed_set._sort_like_sizes(quantities)

    def test_filter_by_mask_single_array(self):
        """Test _filter_by_mask method with single array."""
        sizes = [1, 2, 3, 4]
        quantities = [10, 20, 30, 40]
        indexed_set = IndexedSet(sizes)

        # Filter out elements where quantities > 25
        mask = np.array([True, True, False, False])
        filtered_quantities = indexed_set._filter_by_mask(mask, quantities)

        # Check that sizes were updated
        np.testing.assert_array_equal(indexed_set.sizes, [1, 2])
        # Check that quantities were filtered
        np.testing.assert_array_equal(filtered_quantities, [10, 20])

    def test_filter_by_mask_multiple_arrays(self):
        """Test _filter_by_mask method with multiple arrays."""
        sizes = [1, 2, 3, 4]
        quantities = [10, 20, 30, 40]
        weights = [0.1, 0.2, 0.3, 0.4]
        indexed_set = IndexedSet(sizes)

        mask = np.array([True, False, True, False])
        filtered_quantities, filtered_weights = indexed_set._filter_by_mask(
            mask, quantities, weights
        )

        # Check that sizes were updated
        np.testing.assert_array_equal(indexed_set.sizes, [1, 3])
        # Check that arrays were filtered
        np.testing.assert_array_equal(filtered_quantities, [10, 30])
        np.testing.assert_array_equal(filtered_weights, [0.1, 0.3])

    def test_as_int_method(self):
        """Test _as_int method functionality."""
        sizes = [1, 2, 3]
        indexed_set = IndexedSet(sizes)

        # Test with simple fractions
        lst = [0.1, 0.2, 0.3]
        int_arr = indexed_set._as_int(lst)

        assert isinstance(int_arr, np.ndarray)
        assert int_arr.dtype == int
        # Should find a multiplier that makes all values integers
        assert np.all(int_arr % 1 == 0)
        np.testing.assert_array_equal(int_arr, [1, 2, 3])

    def test_as_int_with_already_integers(self):
        """Test _as_int method with already integer values."""
        sizes = [1, 2, 3]
        indexed_set = IndexedSet(sizes)

        lst = [1, 2, 3]
        int_arr = indexed_set._as_int(lst)

        np.testing.assert_array_equal(int_arr, [1, 2, 3])
