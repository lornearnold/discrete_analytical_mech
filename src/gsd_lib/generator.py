"""MinimalPackingGenerator class for generating minimal packing configurations."""

from fractions import Fraction

import numpy as np

from .sample import Sample
from .utils import rep_size_by_vol


class MinimalPackingGenerator:
    """A class to generate minimal packing configurations for granular materials."""

    _ok_max_size = ("representative", "min", "max", "random")

    def __init__(self, gsd, max_size="representative", order=1, tol=0.0, density=1.0):
        """
        Initialize MinimalPackingGenerator.

        Parameters:
        -----------
        gsd : GSD
            Target grain size distribution
        max_size : str
            Method for determining maximum particle size
        order : int
            Order of approximation for size generation
        tol : float
            Tolerance for error checking
        density : float
            Particle density
        """
        self.target_gsd = gsd
        self.density = density
        self.max_particle_size = self._get_max_particle_size(max_size)
        self.sample_sizes = None
        self._set_sample_sizes(order)
        self.order = order
        self.tol = tol

        self._iterations = 0

        self.kappa = None
        self.min_particles = None
        self.phi_n = None
        self.nu_n = None
        self.xi_n = None
        self.weights = None
        self.particles = None
        self.weight_factor = None
        self.mps = self._get_minimal_packing_set()

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
        _sizes = np.zeros(len(self.target_gsd.sizes) - 1) + self.max_particle_size
        _low_sample_sizes = (
            np.zeros(len(self.target_gsd.sizes) - 1) + self.max_particle_size
        )
        _high_sample_sizes = (
            np.zeros(len(self.target_gsd.sizes) - 1) + self.max_particle_size
        )
        for i in range(len(self.target_gsd.sizes) - 2):
            if order == 0:
                # Randomly distribute sizes within their bins
                _sizes[i] = np.random.uniform(
                    self.target_gsd.sizes[i], self.target_gsd.sizes[i + 1]
                )
            elif order == 1 or order == 2:
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
        # particle_volumes = self.target_gsd._as_int(sphere_vol(_int_bins))
        xi = np.asarray(
            [
                Fraction(numerator=n, denominator=particle_volumes[-1])
                for n in particle_volumes
            ]
        )
        return xi

    def _get_test_sample(self, trial_sizes):
        """
        Generate a test sample with given trial sizes.
        """
        test_sample = Sample(trial_sizes, np.ones(len(trial_sizes), dtype=int))
        self.xi = self._xi_volume_ratios(trial_sizes=trial_sizes)
        phi = self.target_gsd.phi
        kappa = phi / self.xi
        lcm = np.lcm.reduce([fr.denominator for fr in kappa])
        tries = (lcm - 1) * (self.order == 2) + 1

        for i in range(1, tries + 1):
            test_min = np.asarray(
                [
                    max(
                        1,
                        int(
                            np.multiply(fr.numerator, i / fr.denominator, dtype=object)
                        ),
                    )
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
