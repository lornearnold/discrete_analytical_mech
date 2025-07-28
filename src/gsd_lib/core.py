# #
# import warnings
# from fractions import Fraction
# from functools import reduce
# from math import gcd

# import numpy as np


# def check_weights(contents):
#     if sum(contents) != 1:
#         raise ValueError("weights must sum to 1")


# def check_particles(contents):
#     if not all([x > 0 for x in contents]):
#         raise ValueError("particles must be positive numbers")


# def is_sorted(lst):
#     return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


# def sphere_vol(d):
#     # r = d / 2
#     return d**3  # 4 / 3 * np.pi * r**3


# def sphere_rad(v):
#     return (3 * v / (4 * np.pi)) ** (1 / 3)


# def sphere_diam(v):
#     return sphere_rad(v) * 2


# def rep_size_by_vol(d_low, d_high):
#     p = (2 - 4 ** (1 / 3)) * (d_low / d_high) + 4 ** (1 / 3)
#     d_rep = d_low + (d_high - d_low) / p
#     return d_rep


# class IndexedSet:
#     """
#     Base class for indexed sets with sizes.
#     Ensures sizes are unique and sorted, and provides utilities for child classes
#     to maintain attributes that should be indexed with sizes.
#     """

#     def __init__(self, sizes):
#         """
#         Initialize IndexedSet with sizes array.

#         Parameters:
#         -----------
#         sizes : array-like
#             Array of size values that must be unique and will be sorted
#         """
#         # Convert to numpy array
#         sizes = np.asarray(sizes)

#         # Check for unique sizes
#         if len(np.unique(sizes)) != len(sizes):
#             raise ValueError("All entries in sizes must be unique")

#         # Get sorting indices and store them for child classes
#         self._sort_indices = np.argsort(sizes)

#         # Sort sizes
#         self.sizes = sizes[self._sort_indices]

#     def _sort_like_sizes(self, *arrays):
#         """
#         Sort one or more arrays using the same indices that were used to sort sizes.

#         Parameters:
#         -----------
#         *arrays : array-like
#             Arrays to be sorted in the same order as sizes

#         Returns:
#         --------
#         tuple or single array
#             Sorted arrays in the same order as sizes
#         """
#         sorted_arrays = []
#         for arr in arrays:
#             arr = np.asarray(arr)
#             if len(arr) != len(self.sizes):
#                 raise ValueError(
#                     f"Array length {len(arr)} must match sizes length {len(self.sizes)}"
#                 )
#             sorted_arrays.append(arr[self._sort_indices])

#         return sorted_arrays[0] if len(sorted_arrays) == 1 else tuple(sorted_arrays)

#     def _filter_by_mask(self, mask, *arrays):
#         """
#         Filter sizes and associated arrays using a boolean mask.
#         Updates self.sizes and returns filtered arrays.

#         Parameters:
#         -----------
#         mask : array-like of bool
#             Boolean mask to apply
#         *arrays : array-like
#             Arrays to be filtered along with sizes

#         Returns:
#         --------
#         tuple or single array
#             Filtered arrays
#         """
#         self.sizes = self.sizes[mask]

#         filtered_arrays = []
#         for arr in arrays:
#             arr = np.asarray(arr)
#             filtered_arrays.append(arr[mask])

#         return (
#             filtered_arrays[0] if len(filtered_arrays) == 1 else tuple(filtered_arrays)
#         )

#     def __len__(self):
#         """
#         Return the number of unique sizes in the set.
#         """
#         return len(self.sizes)

#     def _as_int(self, lst):
#         lst = np.asarray(lst, dtype=float)

#         # Handle edge cases
#         if len(lst) == 0:
#             return np.array([], dtype=int)
#         if np.all(lst == 0):
#             return np.zeros_like(lst, dtype=int)

#         # Convert to fractions to get exact rational representation
#         fractions = [Fraction(x).limit_denominator() for x in lst]

#         # Find LCM of all denominators
#         denominators = [f.denominator for f in fractions]
#         lcm = denominators[0]
#         for d in denominators[1:]:
#             lcm = lcm * d // gcd(lcm, d)

#         # Scale by LCM to get integers
#         int_array = np.array([int(f * lcm) for f in fractions], dtype=int)

#         # Reduce by GCD to get smallest possible integers
#         if len(int_array) > 1:
#             array_gcd = reduce(gcd, int_array)
#             if array_gcd > 1:
#                 int_array = int_array // array_gcd

#         return int_array

#     # def _as_int(self, lst):
#     #     lst = np.asarray(lst)
#     #     i = 1
#     #     denom = min(1, np.min(lst))
#     #     mult = i / denom
#     #     while True:
#     #         # Vectorized check - much faster for numpy arrays
#     #         if np.all((lst * mult) % 1 == 0):
#     #             break
#     #         i += 1
#     #         mult = i / denom
#     #     int_array = (lst * mult).astype(int)
#     #     return int_array


# class Sample(IndexedSet):
#     """
#     A class representing a sample with particle sizes and associated data.
#     Inherits from IndexedSet to maintain sorted sizes and corresponding data arrays.
#     """

#     def __init__(
#         self,
#         sizes,
#         quantities,
#         shape="sphere",
#         density=1.0,
#     ):
#         """
#         Initialize Sample with particle sizes and associated data.

#         Parameters:
#         -----------
#         sizes : array-like
#             Array of particle size values that must be unique and positive
#         quantities : array-like
#             Array of particle quantity values that must be positive integers.
#         shape : str, optional
#             Shape of particles (default: "sphere")
#         density : float, optional
#             Density of particles (default: 1.0)
#         """
#         # Call parent constructor with just sizes
#         super().__init__(sizes)

#         # Validate that sizes are positive (specific to Sample)
#         if np.any(self.sizes <= 0):
#             raise ValueError("All particle sizes must be positive")

#         self.shape = shape
#         self.density = density

#         # Sort quantities to match sorted sizes
#         quantities = self._sort_like_sizes(quantities)

#         # Validate quantities
#         quantities = np.asarray(quantities)
#         if np.any(quantities < 0):
#             raise ValueError("All quantities must be positive integers")

#         # Handle zero quantities
#         if np.any(quantities == 0):
#             warnings.warn("Zero quantities found. Removing corresponding entries.")
#             non_zero_mask = quantities > 0
#             quantities = self._filter_by_mask(non_zero_mask, quantities)

#         # Handle floating-point quantities
#         if not all(isinstance(q, (int, np.integer)) for q in quantities):
#             warnings.warn("Floating-point quantities found. Converting to integers.")
#             quantities = np.round(quantities).astype(int)

#         self.quantities = quantities
#         self.total_masses = self.quantities * sphere_vol(self.sizes) * self.density

#     def norm_1(self):
#         """
#         Return the L-1 norm of the sample quantities, which is the total number of particles.
#         """
#         return np.sum(self.quantities)


# class GSD(IndexedSet):
#     """
#     class: representing a grain size distribution
#     """

#     _ok_desc = ("contains", "efficiently", "articulately", "accurately")

#     def __init__(
#         self,
#         sizes,
#         percent_retained=None,
#         masses=None,
#     ):
#         """
#         Initialize GSD with sizes and masses.

#         Parameters:
#         -----------
#         sizes : array-like
#             Array of size values that must be unique and will be sorted
#         masses : array-like
#             Array of mass values corresponding to sizes
#         """
#         # Call parent constructor with just sizes
#         super().__init__(sizes)

#         # Validate either percent_retained or masses, but not both, is provided
#         if percent_retained is not None and masses is not None:
#             raise ValueError(
#                 "Either percent_retained or masses must be provided, not both."
#             )

#         if percent_retained is None and masses is None:
#             raise ValueError(
#                 "Either percent_retained or masses must be provided to initialize GSD."
#             )

#         # If percent_retained is provided, convert it to masses
#         if percent_retained is not None:
#             self.percent_retained = np.asarray(percent_retained)
#             self.masses = np.asarray(percent_retained)
#         else:
#             self.masses = np.asarray(masses)
#             self.percent_retained = self.masses / np.sum(self.masses)

#         if self.masses[-1] != 0:
#             warnings.warn(
#                 "This GSD is incomplete: The mass retained on the largest size bin is not zero. Therefore, the upper limit of particle sizes is undefined."
#             )

#         self.percent_passing = 1 - np.cumsum(self.percent_retained)

#         _int_percent_retained = self._as_int(self.percent_retained[:-1])
#         self.phi = np.asarray(
#             [
#                 Fraction(numerator=n, denominator=_int_percent_retained[-1])
#                 for n in _int_percent_retained
#             ]
#         )

#     def describes(self, sample: Sample, how="contains"):
#         """
#         Check whether this GSD meets the formal definition of describing a sample.
#         Returns True if the GSD describes the sample, False otherwise.
#         Parameters:
#         -----------
#         sample : Sample
#             The sample to describe the GSD in relation to.
#         """
#         if how not in GSD._ok_desc:
#             raise ValueError(f"how must be one of {GSD._ok_desc}, got {how}")
#         if not isinstance(sample, Sample):
#             raise TypeError("Sample must be an instance of the Sample class")

#         # TODO: # Implement different methods of describing based on 'how'
#         # For now, we will just check if the GSD contains the sample sizes
#         return (self.sizes[0] < sample.sizes[0]) and (
#             self.sizes[-1] >= sample.sizes[-1]
#         )

#     def description_error(self, sample: Sample):
#         """
#         Calculate the description error of this GSD in relation to a sample.
#         Returns an array representing the error for each bin size.
#         Parameters:
#         -----------
#         sample : Sample
#             The sample to describe the GSD in relation to.
#         """
#         if not isinstance(sample, Sample):
#             raise TypeError("Sample must be an instance of the Sample class")

#         sample_mass = np.sum(sample.total_masses)
#         sample_percent_retained = []

#         for i in range(len(self.sizes) - 1):
#             sample_indices = np.where(
#                 (sample.sizes >= self.sizes[i]) & (sample.sizes < self.sizes[i + 1])
#             )[0]  # TODO: fix greater than or equal to
#             percent_between = np.sum(sample.total_masses[sample_indices]) / sample_mass
#             sample_percent_retained.append(percent_between)

#         retained_on_last = (
#             np.sum(sample.total_masses[sample.sizes > self.sizes[-1]]) / sample_mass
#             if len(sample.sizes) > 0
#             else 0.0
#         )
#         sample_percent_retained.append(retained_on_last)

#         return np.array(sample_percent_retained) - self.percent_retained


# class MinimalPackingGenerator:
#     """A class to generate minimal packing configurations for granular materials."""

#     _ok_max_size = ("representative", "min", "max", "random")

#     def __init__(
#         self, gsd: GSD, max_size="representative", order=1, tol=0.0, density=1.0
#     ):
#         self.target_gsd = gsd
#         self.density = density
#         self.max_particle_size = self._get_max_particle_size(max_size)
#         self.sample_sizes = None
#         self._set_sample_sizes(order)
#         self.order = order
#         self.tol = tol

#         self._int_masses = None
#         self._int_bins = None

#         self._iterations = 0

#         self.kappa = None
#         self.min_particles = None
#         self.phi_n = None
#         self.nu_n = None
#         self.xi_n = None
#         self.weights = None
#         self.particles = None
#         self.int_weights = None
#         self.weight_factor = None
#         self.mps = self._get_minimal_packing_set()

#     def _get_max_particle_size(self, max_size):
#         """
#         Get the maximum size based on the specified max_size type.
#         """
#         if max_size == "representative":
#             return rep_size_by_vol(self.target_gsd.sizes[-2], self.target_gsd.sizes[-1])
#         elif max_size == "min":
#             return self.target_gsd.sizes[-2]
#         elif max_size == "max":
#             return self.target_gsd.sizes[-1]
#         elif max_size == "random":
#             return np.random.uniform(
#                 self.target_gsd.sizes[-2], self.target_gsd.sizes[-1]
#             )
#         else:
#             raise ValueError(
#                 f"max_size must be one of {MinimalPackingGenerator._ok_max_size}"
#             )

#     def _set_sample_sizes(self, order):
#         """
#         Generate sample sizes based on the following orders:
#         0 - the sizes are randomly distributed within their bins
#         1 - the sizes are representative of a uniform distribution within the bins
#         2 - the sizes are representative of a uniform distribution within the bins (and eventually, will search to tolerance)
#         3 - the max and min particle sizes are evaluated to find the best fit
#         """
#         _sizes = np.zeros(len(self.target_gsd.sizes) - 1) + self.max_particle_size
#         _low_sample_sizes = (
#             np.zeros(len(self.target_gsd.sizes) - 1) + self.max_particle_size
#         )
#         _high_sample_sizes = (
#             np.zeros(len(self.target_gsd.sizes) - 1) + self.max_particle_size
#         )
#         for i in range(len(self.target_gsd.sizes) - 2):
#             if order == 0:
#                 # Randomly distribute sizes within their bins
#                 _sizes[i] = np.random.uniform(
#                     self.target_gsd.sizes[i], self.target_gsd.sizes[i + 1]
#                 )
#             elif order == 1 or order == 2:
#                 # Use representative sizes within their bins
#                 _sizes[i] = rep_size_by_vol(
#                     self.target_gsd.sizes[i], self.target_gsd.sizes[i + 1]
#                 )
#                 # _sizes[i] = self.target_gsd.sizes[i]
#             elif order == 3:
#                 # Evaluate max and min particle sizes
#                 _low_sample_sizes[i] = self.target_gsd.sizes[i]
#                 _high_sample_sizes[i] = self.target_gsd.sizes[i + 1]

#         self.sample_sizes = _sizes
#         self._low_sample_sizes = _low_sample_sizes
#         self._high_sample_sizes = _high_sample_sizes

#     def _xi_volume_ratios(self, trial_sizes):
#         """
#         Calculate the volume ratios of the each size relative to the largest size sample for individual particles.
#         Returns:
#         --------
#         np.ndarray
#             Array of volume ratios for each particle size.
#         """
#         _int_bins = self.target_gsd._as_int(trial_sizes)
#         particle_volumes = self.target_gsd._as_int(sphere_vol(_int_bins))
#         xi = np.asarray(
#             [
#                 Fraction(numerator=n, denominator=particle_volumes[-1])
#                 for n in particle_volumes
#             ]
#         )
#         return xi

#     def _get_test_sample(self, trial_sizes):
#         test_sample = Sample(trial_sizes, np.ones(len(trial_sizes), dtype=int))
#         self.xi = self._xi_volume_ratios(trial_sizes=trial_sizes)
#         phi = self.target_gsd.phi
#         kappa = phi / self.xi
#         lcm = np.lcm.reduce([fr.denominator for fr in kappa])
#         tries = (lcm - 1) * (self.order == 2) + 1

#         for i in range(1, tries + 1):
#             test_min = np.asarray(
#                 [
#                     max(
#                         1,
#                         int(
#                             np.multiply(fr.numerator, i / fr.denominator, dtype=object)
#                         ),
#                     )
#                     for fr in kappa
#                 ]
#             )
#             test_sample = Sample(trial_sizes, test_min)
#             test_errors = self.target_gsd.description_error(test_sample)
#             if all([abs(x) < self.tol for x in test_errors]):
#                 break

#         return test_sample

#     def _get_minimal_packing_set(self):
#         """
#         Generate a minimal packing set based on the target GSD and sample sizes.
#         """
#         if self.order != 3:
#             trial_sizes = self.sample_sizes
#         else:
#             _low_size_q = self._get_test_sample(self._low_sample_sizes).quantities
#             _high_size_q = self._get_test_sample(self._high_sample_sizes).quantities

#             xi = self.target_gsd.phi[:-1] / _high_size_q
#             size_ratios = xi ** (1 / 3)
#             trial_sizes = size_ratios * self.max_particle_size
#         mps = self._get_test_sample(trial_sizes)
#         return mps


# if __name__ == "__main__":
#     # weight, r = gsd_gen(4, size_ratio=10)
#     r = [0.111, 0.222, 0.333, 0.444, 0.555]
#     weight = [0.39, 0.20, 0.14, 0.27, 0.0]
#     g = GSD(sizes=r, masses=weight)
#     s = MinimalPackingGenerator(g, max_size="representative", order=2, tol=0.000001)
#     print(f"Errors: {g.description_error(s.mps)}")
#     print(f"Sample quantities: {s.mps.quantities}")
