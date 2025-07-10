"""MinimalPackingGenerator class for generating minimal packing configurations."""

import numpy as np

from .sample import Sample


class MinimalPackingGenerator:
    """A class to generate minimal packing configurations for granular materials."""

    _ok_x_n_factor = (0.0, 1.0)

    def __init__(self, gsd, x_n_factor=0.5, tol=1e-3, flex=False, density=1.0):
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
        self.g = gsd
        self.density = density
        self.tol = tol

        self._iteration = 1

        self.card_s = len(self.g) - 1
        self.x_n_factor = x_n_factor
        self.x_plus = self._get_x_s(extreme=1)
        self.x_minus = self._get_x_s(extreme=0)
        self.phi = self._get_phi()
        self.zeta_plus = self._get_zeta(self.x_plus)
        self.zeta_minus = self._get_zeta(self.x_minus)
        self.kappa_plus = self.phi / self.zeta_plus
        self.kappa_minus = self.phi / self.zeta_minus

        self.qs = []
        self.q_min_max_ratios = []
        self.errors = []

        if flex:
            self.mps: Sample = self._get_minimal_packing_set()
        else:
            self.mps: Sample = self._get_constrained_packing_set()

    def _get_x_s(self, extreme: int):
        """
        Get the maximum size based on the specified max_size type.
        """
        factors = extreme * np.ones(self.card_s)
        factors[-1] = self.x_n_factor

        x = self.g.sizes[:-1] + factors * (self.g.sizes[1:] - self.g.sizes[:-1])
        # if extreme == 0:
        #     x[:-1] = np.nextafter(x[:-1], x[1:])
        return x

    def _get_phi(self):
        """
        Calculate the phi values for the sample sizes.
        """
        if self.g.percent_retained[-1] != 0:
            raise Warning(
                "The last percent_retained value is not zero, indicating an incomplete GSD."
            )

        m = self.g.percent_retained[:-1]
        m_ns = m[-1]
        phi = m / m_ns  # TEST: phi[-1] should always be 1.0
        return phi

    def _get_zeta(self, x):
        """
        Calculate the zeta values for the sample sizes.

        """
        v = x**3
        v_ns = v[-1]
        zeta = v / v_ns  # TEST: zeta[-1] should always be 1.0
        return zeta

    # def _get_test_sample(self, x, kappa, i):
    #     """
    #     Generate a test sample with given trial sizes.
    #     """
    #     test_qs = max(1, int(kappa * i))
    #     test_sample = Sample(x, test_qs)

    #     return test_sample

    def _int_between(self, i):
        """
        Generate an array of integers that's the smallest integer between two arrays of floats.
        If no such integer exists, return the next highest integer above the smaller value.
        Return the integer array and a boolean indicating if an integer between each pair exists.
        """
        # Note: the _plus and _minus denote whether the sample size is skewed to the largest or smallest possible size.
        # Smaller particles will mean larger number of particles, so kappa_minus will be larger than kappa_plus.
        q_minus = np.floor(self.kappa_minus * i).astype(int)
        q_plus = np.ceil(self.kappa_plus * i).astype(int)

        int_exists = np.all(q_minus >= q_plus)
        int_array = q_plus

        return int_array, int_exists

    def _get_best_size(self, q_int, i):
        """
        Generate the best sizes for a test sample based on the integer quantities.
        """
        new_zeta = self.phi / (q_int / i)
        best_sizes = new_zeta ** (1 / 3) * self.x_plus[-1]
        # if any of the sizes are less than x_minus or greater than x_plus, replace them with the bounds
        best_sizes = np.clip(best_sizes, self.x_minus, self.x_plus)
        return best_sizes

    def _get_minimal_packing_set(self):
        """
        Generate a minimal packing set based on the target GSD and sample sizes.
        """
        s_i = Sample([1], [1])

        for i in range(1, 1000):
            q_int, stop = self._int_between(i)
            sizes = self._get_best_size(q_int, i)
            s_i = Sample(sizes, q_int)
            self.qs.append(q_int)
            self.errors.append(self.g.description_error(s_i))
            if stop:
                self._iteration = i
                break

        return s_i

    def _get_constrained_packing_set(self):
        """
        Generate a minimal packing set based on the target GSD and sample sizes.
        """
        s_i = Sample([1], [1])

        for i in range(1, 10000):
            q_int = np.ceil(self.kappa_plus * i).astype(int)
            # sizes = self._get_best_size(q_int, i)
            s_i = Sample(self.x_plus, q_int)
            self.qs.append(q_int)
            err = self.g.description_error(s_i)
            self.errors.append(err)
            self._iteration = i
            if np.all(err <= self.tol):
                break

        return s_i
