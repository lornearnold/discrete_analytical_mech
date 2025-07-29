"""GSD (Grain Size Distribution) class."""

import warnings

import numpy as np
from scipy.interpolate import CubicSpline

from .base import IndexedSet
from .uscs import Simple_sample, classify_uscs


def add_boundary_knots(spline):
    """
    # Copied from https://docs.scipy.org/doc/scipy/tutorial/interpolate/extrapolation_examples.html#cubicspline-extend-the-boundary-conditions
    Add knots infinitesimally to the left and right.

    Additional intervals are added to have zero 2nd and 3rd derivatives,
    and to maintain the first derivative from whatever boundary condition
    was selected. The spline is modified in place.
    """
    # determine the slope at the left edge
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)

    # add a new breakpoint just to the left and use the
    # known slope to construct the PPoly coefficients.
    leftxnext = np.nextafter(leftx, leftx - 1)
    leftynext = lefty + leftslope * (leftxnext - leftx)
    leftcoeffs = np.array([0, 0, leftslope, leftynext])
    spline.extend(leftcoeffs[..., None], np.r_[leftxnext])

    # repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx, nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.array([0, 0, rightslope, rightynext])
    spline.extend(rightcoeffs[..., None], np.r_[rightxnext])


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
        percent_retained : array-like, optional
            Array of percent retained values corresponding to sizes
        masses : array-like, optional
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

        if abs(np.sum(self.percent_retained) - 1) > 1e-6:
            raise ValueError(
                "The sum of percent_retained must equal 1.0 (within 1e-6). "
                f"Current sum: {np.sum(self.percent_retained)}"
            )

        if self.masses[-1] != 0:
            warnings.warn(
                "This GSD is incomplete: The mass retained on the largest size bin is not zero. Therefore, the upper limit of particle sizes is undefined."
            )

    def describes(self, sample, how="contains"):
        """
        Check whether this GSD meets the formal definition of describing a sample.
        Returns True if the GSD describes the sample, False otherwise.
        Parameters:
        -----------
        sample : Sample
            The sample to describe the GSD in relation to.
        how : str
            Method of description checking
        """
        from .sample import Sample  # Import here to avoid circular import

        if how not in GSD._ok_desc:
            raise ValueError(f"how must be one of {GSD._ok_desc}, got {how}")
        if not isinstance(sample, Sample):
            raise TypeError("Sample must be an instance of the Sample class")

        # TODO: # Implement different methods of describing based on 'how'
        # For now, we will just check if the GSD contains the sample sizes
        return (self.sizes[0] < sample.sizes[0]) and (
            self.sizes[-1] >= sample.sizes[-1]
        )

    def description_error(self, sample):
        """
        Calculate the description error of this GSD in relation to a sample.
        Returns an array representing the error for each bin size.
        Parameters:
        -----------
        sample : Sample
            The sample to describe the GSD in relation to.
        """
        from .sample import Sample  # Import here to avoid circular import

        if not isinstance(sample, Sample):
            raise TypeError("Sample must be an instance of the Sample class")

        sample_mass = np.sum(sample.total_masses)
        sample_percent_retained = []

        for i in range(len(self.sizes) - 1):
            sample_indices = np.where(
                (sample.sizes > self.sizes[i])
                & (np.nextafter(sample.sizes, self.sizes[i]) <= self.sizes[i + 1])
            )[0]
            percent_between = np.sum(sample.total_masses[sample_indices]) / sample_mass
            sample_percent_retained.append(percent_between)

        retained_on_last = (
            np.sum(sample.total_masses[sample.sizes > self.sizes[-1]]) / sample_mass
            if len(sample.sizes) > 0
            else 0.0
        )
        sample_percent_retained.append(retained_on_last)

        return np.array(sample_percent_retained) - self.percent_retained

    def _i_gs_curve(self, lower_bound=0.001, upper_bound=75.0):
        """
        Extend the GSD curve over tha required range for defining the grain-size index, I_GS as described by (Erguler, 2016).
        The range is from 0.001 to 75 mm."""
        size_points = self.sizes
        gsd_curve = self.percent_passing

        # These commented out lines complete I_GS, but I'm turning them off for now.
        # This makes a modified I_GS that is more appropriate for the MPS work I'm doing right now.
        if size_points[-1] < upper_bound:
            size_points = np.insert(size_points, len(size_points), upper_bound)
            gsd_curve = np.insert(gsd_curve, len(gsd_curve), gsd_curve[-1])
            # size_points = np.concatenate(
            #     [size_points, np.linspace(size_points[-1], right_bound, n_extend)]
            # )
            # gsd_curve = np.concatenate([gsd_curve, np.ones(n_extend) * gsd_curve[-1]])

        if size_points[0] > lower_bound:
            size_points = np.insert(size_points, 0, lower_bound)
            gsd_curve = np.insert(gsd_curve, 0, gsd_curve[0])

        if size_points[0] < lower_bound:
            gs_min = np.interp(np.log10(lower_bound), np.log10(size_points), gsd_curve)
            size_points[0] = lower_bound
            gsd_curve[0] = gs_min
            # spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
            # add_boundary_knots(spline)
            # log_size = np.linspace(np.log10(lower_bound), log_size[-1], 100)
            # gsd_curve = spline(log_size, nu=0)

        return np.log10(size_points), gsd_curve

    def _percent_in_range(self, range: tuple):
        """
        Calculate the percent of the GSD that falls within a specified range.

        Parameters:
        -----------
        range : tuple
            A tuple specifying the lower and upper bounds of the range (lower, upper).

        Returns:
        --------
        float
            The percent of the GSD that falls within the specified range.
        """
        if len(self.sizes) == 0:
            return 0.0

        lower_bound, upper_bound = range
        mask = (self.sizes >= lower_bound) & (self.sizes < upper_bound)
        return np.sum(self.percent_retained[mask])

    @property
    def percent_fines(self):
        """
        Calculate the percent of fines in the GSD.

        Returns:
        --------
        float
            The percent of fines in the GSD.
        """
        return self._percent_in_range((0, 0.075))

    @property
    def percent_sand(self):
        """
        Calculate the percent of sand in the GSD.

        Returns:
        --------
        float
            The percent of sand in the GSD.
        """
        return self._percent_in_range((0.075, 4.75))

    @property
    def percent_gravel(self):
        """
        Calculate the percent of gravel in the GSD.

        Returns:
        --------
        float
            The percent of gravel in the GSD.
        """
        return self._percent_in_range((4.75, 75.0))

    @property
    def curvature(self):
        """
        Calculate the curvature of the GSD curve.

        Returns:
        --------
        float
            Curvature of the GSD curve, defined as the maximum curvature divided by the range of sizes
        """
        log_size = np.log10(self.sizes)
        gsd_curve = self.percent_passing
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        second_derivative = spline(log_size, nu=2)
        return second_derivative

    @property
    def slope(self):
        """
        Calculate the maximum slope of the GSD curve.

        Returns:
        --------
        float
            Maximum slope of the GSD curve
        """
        log_size = np.log10(self.sizes)
        gsd_curve = self.percent_passing
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        first_derivative = spline(log_size, nu=1)
        return first_derivative

    @property
    def gs_index(self):
        """
        Calculate the grain-size index (I_GS) as described by Erguler (2016).
        """
        log_size, gsd_curve = self._i_gs_curve()
        a_t = log_size[-1] - log_size[0]
        a_c = np.trapezoid(gsd_curve, log_size)
        return a_c / a_t

    @property
    def curvature_index(self):
        """
        Calculate the grain-size index (I_GS) modified to describe only the grain sieve sizes included in the GSD.
        """
        log_size = np.log10(self.sizes)
        # End points for slope calculation:
        a = 1
        b = -1
        base = log_size[b] - log_size[a]
        slope = (self.percent_passing[b] - self.percent_passing[a]) / base

        uncurved_area = (0 * self.percent_passing[a] + (slope / 2)) * base**2
        area = (
            np.trapezoid(self.percent_passing[a:], log_size[a:])
            - base * self.percent_passing[a]
        )
        return area / uncurved_area

    def _d_percent(self, percent, suppress_warnings=False):
        """
        Find the size corresponding to a given percent passing.
        Interpolate on a semilog-x scale.
        """
        log_size = np.log10(self.sizes)
        gsd_curve = self.percent_passing

        if percent < self.percent_passing[0] and not suppress_warnings:
            warnings.warn(
                f"The minimum size has more than {percent * 100:.1f}% passing. "
                f"Extrapolating below the minimum size.",
                UserWarning,
            )

            spline = CubicSpline(self.percent_passing, log_size, bc_type="natural")
            # extend the natural natural spline with linear extrapolating knots
            add_boundary_knots(spline)
            log_size = np.linspace(0.0001, log_size[-1], 1000)
            gsd_curve = spline(log_size, nu=0)

        log_d = np.interp(percent, gsd_curve, log_size)
        return 10**log_d

    @property
    def d_10(self):
        """
        Calculate D10 (size at 10% passing).

        Returns:
        --------
        float
            Size corresponding to 10% passing
        """
        return self._d_percent(0.10)

    @property
    def d_30(self):
        """
        Calculate D30 (size at 30% passing).

        Returns:
        --------
        float
            Size corresponding to 30% passing
        """
        return self._d_percent(0.30)

    @property
    def d_60(self):
        """
        Calculate D60 (size at 60% passing).

        Returns:
        --------
        float
            Size corresponding to 60% passing
        """
        return self._d_percent(0.60)

    @property
    def cu(self):
        """
        Calculate uniformity coefficient (Cu).

        Returns:
        --------
        float
            Uniformity coefficient, defined as D60 / D10
        """
        return self.d_60 / self.d_10 if self.d_10 > 0 else np.inf

    @property
    def cc(self):
        """
        Calculate coefficient of curvature (Cc).

        Returns:
        --------
        float
            Coefficient of curvature, defined as (D30^2) / (D60 * D10)
        """

        return (
            (self.d_30**2) / (self.d_60 * self.d_10)
            if self.d_60 > 0 and self.d_10 > 0
            else 0
        )

    @property
    def percent_passing(self):
        """
        Calculate percent passing for each size.

        Returns:
        --------
        np.ndarray
            Array of percent passing values corresponding to sizes
        """
        reverse_cumulative_retained = np.cumsum(self.percent_retained[::-1])[::-1]
        return 1.0 - reverse_cumulative_retained

    @property
    def uscs_symbol(self):
        """
        Classify this GSD according to USCS standards.

        Returns:
        --------
        str
            USCS classification symbol
        """
        sample = Simple_sample(
            gravel=self.percent_gravel,
            sand=self.percent_sand,
            fines=self.percent_fines,
            cu=self.cu,
            cc=self.cc,
        )
        classify_uscs(sample)
        return sample.group_symbol

    @property
    def uscs_name(self):
        """
        Get the USCS classification name.

        Returns:
        --------
        str
            USCS classification name
        """
        sample = Simple_sample(
            gravel=self.percent_gravel,
            sand=self.percent_sand,
            fines=self.percent_fines,
            cu=self.cu,
            cc=self.cc,
        )
        classify_uscs(sample)
        return sample.group_name
