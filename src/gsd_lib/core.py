#
import datetime
import warnings

import numpy as np


def check_weights(contents):
    if sum(contents) != 1:
        raise ValueError("weights must sum to 1")


def check_particles(contents):
    if not all([x > 0 for x in contents]):
        raise ValueError("particles must be positive numbers")


def is_sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


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

    def _as_int(self, lst):
        lst = np.asarray(lst)  # Ensure it's a numpy array
        i = 1
        denom = min(1, np.min(lst))
        mult = i / denom
        while True:
            # Vectorized check - much faster for numpy arrays
            if np.all((lst * mult) % 1 == 0):
                break
            i += 1
            mult = i / denom
        int_array = (lst * mult).astype(int)
        return int_array


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
        self.total_masses = self.quantities * sphere_vol(self.sizes) * self.density

    def norm_1(self):
        """
        Return the L-1 norm of the sample quantities, which is the total number of particles.
        """
        return np.sum(self.quantities)


class GSD(IndexedSet):
    """
    class: representing a grain size distribution
    """

    _ok_desc = ("contains", "efficiently", "articulately", "accurately")

    def __init__(
        self,
        sizes,
        percent_retained=None,
        masses=None,
    ):
        """
        Initialize GSD with sizes and masses.

        Parameters:
        -----------
        sizes : array-like
            Array of size values that must be unique and will be sorted
        masses : array-like
            Array of mass values corresponding to sizes
        """
        # Call parent constructor with just sizes
        super().__init__(sizes)

        # Validate either percent_retained or masses, but not both, is provided
        if percent_retained is not None and masses is not None:
            raise ValueError(
                "Either percent_retained or masses must be provided, not both."
            )

        if percent_retained is None and masses is None:
            raise ValueError(
                "Either percent_retained or masses must be provided to initialize GSD."
            )

        # If percent_retained is provided, convert it to masses
        if percent_retained is not None:
            self.percent_retained = np.asarray(percent_retained)
            self.masses = np.asarray(percent_retained)
        else:
            self.masses = np.asarray(masses)
            self.percent_retained = self.masses / np.sum(self.masses)

        if self.masses[-1] != 0:
            warnings.warn(
                "This GSD is incomplete: The mass retained on the largest size bin is not zero. Therefore, the upper limit of particle sizes is undefined."
            )

        self.phi = self.percent_retained / self.percent_retained[-2]

    def describes(self, sample: Sample, how="contains"):
        """
        Check whether this GSD meets the formal definition of describing a sample.
        Returns True if the GSD describes the sample, False otherwise.
        Parameters:
        -----------
        sample : Sample
            The sample to describe the GSD in relation to.
        """
        if how not in GSD._ok_desc:
            raise ValueError(f"how must be one of {GSD._ok_desc}, got {how}")
        if not isinstance(sample, Sample):
            raise TypeError("Sample must be an instance of the Sample class")

        # TODO: # Implement different methods of describing based on 'how'
        # For now, we will just check if the GSD contains the sample sizes
        return (self.sizes[0] < sample.sizes[0]) and (
            self.sizes[-1] >= sample.sizes[-1]
        )

    def description_error(self, sample: Sample):
        """
        Calculate the description error of this GSD in relation to a sample.
        Returns an array representing the error for each bin size.
        Parameters:
        -----------
        sample : Sample
            The sample to describe the GSD in relation to.
        """
        if not isinstance(sample, Sample):
            raise TypeError("Sample must be an instance of the Sample class")

        sample_mass = np.sum(sample.total_masses)
        sample_percent_retained = []

        for i in range(len(self.sizes) - 1):
            sample_indices = np.where(
                (sample.sizes > self.sizes[i]) & (sample.sizes <= self.sizes[i + 1])
            )[0]
            percent_between = np.sum(sample.total_masses[sample_indices]) / sample_mass
            sample_percent_retained.append(percent_between)

        return np.array(sample_percent_retained) - self.percent_retained


class MinimalPackingGenerator:
    """A class to generate minimal packing configurations for granular materials."""

    _ok_max_size = ("representative", "min", "max", "random")

    def __init__(
        self, gsd: GSD, max_size="representative", order=3, tol=0.0, density=1.0
    ):
        self.target_gsd = gsd
        self.density = density
        self.max_particle_size = self._get_max_particle_size(max_size)
        self.sample_sizes = None
        self._set_sample_sizes(order)
        self.order = order
        self.tol = tol

        self._int_masses = None
        self._int_bins = self.target_gsd._as_int(self.target_gsd.sizes)

        self._iterations = 0

        self.kappa = None
        self.min_particles = None
        self.phi_n = None
        self.nu_n = None
        self.xi_n = None
        self.order = None
        self.orders = None
        self.weights = None
        self.particles = None
        self.int_weights = None
        self.weight_factor = None

    def _get_max_particle_size(self, max_size):
        """
        Get the maximum size based on the specified max_size type.
        """
        if max_size == "representative":
            return rep_size_by_vol(self.target_gsd.sizes[-2], self.target_gsd.sizes[-1])
        elif max_size == "min":
            return self.target_gsd.sizes[-2]
        elif max_size == "max":
            return self.target_gsd.sizes[-1]
        elif max_size == "random":
            return np.random.uniform(
                self.target_gsd.sizes[-2], self.target_gsd.sizes[-1]
            )
        else:
            raise ValueError(
                f"max_size must be one of {MinimalPackingGenerator._ok_max_size}"
            )

    def _set_sample_sizes(self, order):
        """
        Generate sample sizes based on the following orders:
        0 - the sizes are randomly distributed within their bins
        1 - the sizes are representative of a uniform distribution within the bins
        2 - the sizes are representative of a uniform distribution within the bins (and eventually, will search to tolerance)
        3 - the max and min particle sizes are evaluated to find the best fit
        """
        _sizes = np.zeros(len(self.target_gsd.sizes) - 1)
        _low_sample_sizes = np.zeros(len(self.target_gsd.sizes) - 1)
        _high_sample_sizes = np.zeros(len(self.target_gsd.sizes) - 1)
        for i in range(len(self.target_gsd.sizes) - 1):
            if order == 0:
                # Randomly distribute sizes within their bins
                _sizes[i] = np.random.uniform(
                    self.target_gsd.sizes[i], self.target_gsd.sizes[i + 1]
                )
            elif order == 1 | order == 2:
                # Use representative sizes within their bins
                _sizes[i] = rep_size_by_vol(
                    self.target_gsd.sizes[i], self.target_gsd.sizes[i + 1]
                )
            elif order == 3:
                # Evaluate max and min particle sizes
                _low_sample_sizes[i] = self.target_gsd.sizes[i]
                _high_sample_sizes[i] = self.target_gsd.sizes[i + 1]

        self.sample_sizes = _sizes
        self._low_sample_sizes = _low_sample_sizes
        self._high_sample_sizes = _high_sample_sizes

    def _xi_volume_ratios(self, trial_sizes):
        """
        Calculate the volume ratios of the each size relative to the largest size sample for individual particles.
        Returns:
        --------
        np.ndarray
            Array of volume ratios for each particle size.
        """
        particle_volumes = sphere_vol(trial_sizes)
        return particle_volumes / particle_volumes[-1]

    def _get_test_sample(self, trial_sizes):
        test_sample = Sample(trial_sizes, np.zeros(len(trial_sizes), dtype=int))
        xi = self._xi_volume_ratios(trial_sizes=trial_sizes)
        phi = self.target_gsd.phi
        kappa = xi / phi
        lcm = np.lcm.reduce([fr.denominator for fr in kappa])
        tries = (lcm - 1) * (self.order == 2) + 1

        for i in range(1, tries + 1):
            test_min = np.asarray(
                [
                    int(np.multiply(fr.numerator, i / fr.denominator, dtype=object))
                    for fr in kappa
                ]
            )
            test_sample = Sample(trial_sizes, test_min)
            test_errors = self.target_gsd.description_error(test_sample)
            if all([abs(x) < self.tol for x in test_errors]):
                break

        return test_sample

    def _get_minimal_packing_set(self):
        """
        Generate a minimal packing set based on the target GSD and sample sizes.
        """
        if self.order != 3:
            trial_sizes = self.sample_sizes
        else:
            _low_size_q = self._get_test_sample(self._low_sample_sizes).quantities
            _high_size_q = self._get_test_sample(self._high_sample_sizes).quantities

            xi = self.target_gsd.phi[:-1] / _high_size_q
            size_ratios = xi ** (1 / 3)
            trial_sizes = size_ratios * self.max_particle_size
        mps = self._get_test_sample(trial_sizes)
        return mps


class GSD_old(IndexedSet):
    """
    class: representing a grain size distribution
    """

    _last_id = int(datetime.datetime.now().timestamp() * 1000)
    _ok_objects = (list, tuple, np.ndarray)
    _ok_types = (int, float, np.int64, np.float64, np.int32, np.float32)
    _ok_cont = ("weights", "particles")

    def __init__(
        self,
        sizes=None,
        contents=None,
        cont_type=_ok_cont[0],
        tol=0.0,
    ):
        super().__init__(sizes)

        # validate input
        if cont_type not in GSD_old._ok_cont:
            raise ValueError(f"content_type must be one of {GSD_old._ok_cont}")
        if sizes is not None:
            sizes = self.check_bins(sizes)

        self.bins = self.sizes  # Use sorted sizes from parent
        self.n_bins = len(self.bins)
        self.int_bins, self.bin_factor = self._as_int(self.bins)

        if contents is not None:
            contents = self._sort_like_sizes(contents)  # Sort contents to match sizes
            contents = self.check_contents(contents, cont_type)

        GSD_old._last_id += 1
        self.id = GSD_old._last_id

        self.min_particles = None
        self.phi_n = None
        self.nu_n = None
        self.xi_n = None
        self.order = None
        self.orders = None
        self.weights = None
        self.particles = None
        self.int_weights = None
        self.weight_factor = None

        self.tol = tol

        if cont_type == self._ok_cont[0]:  # weights
            self.weights = np.asarray(contents)
            self.int_weights, self.weight_factor = self._as_int(self.weights)
            self.get_min_particles()
        if cont_type == self._ok_cont[1]:  # particles
            self.particles = np.asarray(contents)
            self.weights = self.get_weights()

    def __str__(self):
        return f"GSD(id={self.id}, bins={self.bins})"

    # also make a __repr__ method that returns a string that can be used to recreate the object
    # and a __eq__ method that returns True if two objects are equal and False otherwise
    # and a __format__ method that returns a string representation of the object

    def check_bins(self, bins):
        if not isinstance(bins, self._ok_objects):
            raise ValueError(f"bins must be one of {self._ok_objects}")
        if not all(isinstance(item, self._ok_types) for item in bins):
            raise ValueError(f"bins must only contain one of {self._ok_types}")
        if not all([x > 0 for x in bins]):
            raise ValueError("bin values must be positive numbers")
        if not is_sorted(bins):
            warnings.warn("bins are not in ascending order. they will be sorted")
            bins = np.sort(bins)
        return bins

    def check_contents(self, contents, content_type):
        if not isinstance(contents, self._ok_objects):
            raise ValueError(f"{content_type} must be one of {self._ok_objects}")
        if not all(isinstance(item, self._ok_types) for item in contents):
            raise ValueError(
                f"{content_type} must only contain one of {self._ok_types}"
            )
        if len(contents) != self.n_bins:
            raise ValueError(f"{content_type} must have the same length as bins")
        if content_type == self._ok_cont[0]:
            check_weights(contents)

        if content_type == self._ok_cont[1]:
            check_particles(contents)
        return contents

    def get_weights(self, particles=None):
        if particles is None:
            if self.particles is None:
                raise ValueError("Particles must be defined to calculate weights")
            particles = self.particles

        weights = np.asarray([part * rad**3 for part, rad in zip(particles, self.bins)])
        total_weight = np.sum(weights)
        weights = weights / total_weight
        # self.int_weights, self.weight_factor = self._as_int(self.weights)
        # self.get_min_particles()
        return weights


def gsd_gen(n, size_ratio: int = 10, gaps_OK=False, min_r=1, int_sizes=True):
    """
    Generate a grain size distribution (GSD) with n bins and a range of n orders of magnitude.
    :param n: int, number of bins
    :param rnr: int, ratio of largest to smallest bin
    :param gaps_OK: bool, allow gaps in the distribution
    :return: tuple, (percentages, radii)
    """
    # confirm size_ratio is a positive integer
    if not isinstance(size_ratio, int) or size_ratio <= 0:
        raise ValueError("size_ratio must be a positive integer")

    # create a random number generator
    rng = np.random.default_rng()

    # create a list of possible sizes... the size domain
    option_number = min_r * size_ratio - min_r + 1
    if n > size_ratio:
        int_sizes = False
        option_number = n

    sizes = np.linspace(min_r, min_r * size_ratio, option_number, endpoint=True)
    #
    # ## the sizes should be integers if n is less than size_ratio. otherwise floats will be needed
    # for i in range(1, size_ratio + 1):
    #     sizes = np.append(sizes, np.linspace(10 ** (i - 1), 10 ** i, 10 - 1, endpoint=False))
    # if gaps_OK:
    #     p_min = 0
    # # p = rng.integers(low=p_min,high=p_max,size=n)

    # create a random distribution of percentages that sum to 100
    p = rng.dirichlet(np.ones(n), size=1)[0]
    p = np.round(p, 2)
    while True:
        if not gaps_OK:
            p = np.asarray([x + (x == 0) for x in p])
        p_check = 1 - np.sum(p)
        if p_check == 0:
            break
        p[rng.integers(low=0, high=n)] += p_check

    # randomly select n sizes from the list of sizes and sort them in descending order
    r = np.sort(rng.choice(sizes[1::], n - 1, replace=False))
    r = np.append(np.asarray([1]), r)
    if int_sizes:
        r = np.round(r).astype(int)
    return p, r


def sphere_vol(d):
    r = d / 2
    return 4 / 3 * np.pi * r**3


def sphere_rad(v):
    return (3 * v / (4 * np.pi)) ** (1 / 3)


def sphere_diam(v):
    return sphere_rad(v) * 2


def rep_size_by_vol(d_low, d_high):
    p = (2 - 4 ** (1 / 3)) * (d_low / d_high) + 4 ** (1 / 3)
    d_rep = d_low + (d_high - d_low) / p
    return d_rep


if __name__ == "__main__":
    # weight, r = gsd_gen(4, size_ratio=10)
    r = [0.111, 0.222, 0.333, 0.444]
    weight = [0.39, 0.20, 0.14, 0.27]
    particle = [np.int64(92), np.int64(5), np.int64(1), np.int64(1)]
    d = GSD_old(sizes=r, contents=weight, cont_type="weights", tol=0.0000001)
    # d.get_min_particles()
    print(weight, r, d.id, d.min_particles, d.order, d.orders)

    # p, r = gsd_gen(10, size_ratio=1000)
    # c = np.cumsum(p)
    # plt.close('all')
    # plt.plot(r, c)
    # plt.show()

    # bins = [5] #np.linspace(2, 10, 9)
    # ratios = [10, 20, 30, 40, 50]
    # for ratio in ratios:
    #     for n_bins in bins:
    #         percentages, radii = gsd_gen(int(n_bins), size_ratio=ratio)
    #         n = np.asarray(gsd_n(percentages, radii))
    #         o = np.floor(np.log10(n.astype(float)))
    #         ot = np.floor(np.log10(sum(n.astype(float))))
    #         # print(percentages,radii, n)
    #         print(f"With size_ratio = {ratio}, {n_bins} bins, and P_1 = {percentages[0]},"
    #               f" the order of N_large and N_total are {o[-1]} and {ot}")
