#
import datetime
import warnings
from fractions import Fraction

import numpy as np


def check_weights(contents):
    if sum(contents) != 1:
        raise ValueError("weights must sum to 1")


def check_particles(contents):
    if not all([x > 0 for x in contents]):
        raise ValueError("particles must be positive numbers")


def is_sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


class GSD:
    """
    class: representing a grain size distribution
    """

    _last_id = int(datetime.datetime.now().timestamp() * 1000)
    _ok_objects = (list, tuple, np.ndarray)
    _ok_types = (int, float, np.int64, np.float64, np.int32, np.float32)
    _ok_cont = ("weights", "particles")

    def __init__(self, bins=None, contents=None, cont_type=_ok_cont[0], tol=0):
        # validate input
        if cont_type not in GSD._ok_cont:
            raise ValueError(f"content_type must be one of {GSD._ok_cont}")
        if bins is not None:
            bins = self.check_bins(bins)

        self.bins = np.asarray(bins)
        self.n_bins = len(self.bins)
        self.int_bins, self.bin_factor = self._as_int(self.bins)

        if contents is not None:
            contents = self.check_contents(contents, cont_type)

        GSD._last_id += 1
        self.id = GSD._last_id

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

    def get_min_particles(self):
        if self.weights is None:
            raise ValueError("Weights must be defined to calculate number of particles")
        self.phi_n = np.asarray(
            [
                Fraction(numerator=self.int_weights[-1], denominator=d)
                for d in self.int_weights
            ]
        )
        self.xi_n = np.asarray(
            [
                Fraction(numerator=self.int_bins[-1] ** 3, denominator=d**3)
                for d in self.int_bins
            ]
        )
        self.nu_n = self.xi_n / self.phi_n
        lcm = np.lcm.reduce([fr.denominator for fr in self.nu_n])
        if self.tol != 0:
            for i in range(1, lcm):
                test_min = np.asarray(
                    [
                        int(np.multiply(fr.numerator, i / fr.denominator, dtype=object))
                        for fr in self.nu_n
                    ]
                )
                test_weights = self.get_weights(test_min)
                test_errors = (test_weights - self.weights) / self.weights
                total_error = sum(test_errors)
                if all([abs(x) < self.tol for x in test_errors]):
                    self.min_particles = test_min
                    break
            self.min_particles = np.asarray(
                [
                    int(np.multiply(fr.numerator, lcm / fr.denominator, dtype=object))
                    for fr in self.nu_n
                ]
            )
        else:
            self.min_particles = np.asarray(
                [
                    int(np.multiply(fr.numerator, lcm / fr.denominator, dtype=object))
                    for fr in self.nu_n
                ]
            )

        self.orders = np.floor(np.log10(self.min_particles.astype(float)))
        self.order = np.floor(np.log10(sum(self.min_particles.astype(float))))
        if self.particles is None:
            self.particles = self.min_particles
        pass

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

    def _as_int(self, lst):
        i = 1
        denom = min(1, min(lst))
        mult = i / denom
        while True:
            if all([x % 1 == 0 for x in lst * mult]):
                break
            i += 1
            mult = i / denom
        int_bins = (lst * mult).astype(int)
        return int_bins, i


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


def sphere_vol(r):
    return 4 / 3 * np.pi * r**3


def sphere_rad(v):
    return (3 * v / (4 * np.pi)) ** (1 / 3)


if __name__ == "__main__":
    # weight, r = gsd_gen(4, size_ratio=10)
    r = [0.111, 0.222, 0.333, 0.444]
    weight = [0.39, 0.20, 0.14, 0.27]
    particle = [np.int64(92), np.int64(5), np.int64(1), np.int64(1)]
    d = GSD(bins=r, contents=weight, cont_type="weights", tol=0.0000001)
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
