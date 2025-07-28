"""GSD (Grain Size Distribution) class."""

import warnings

import numpy as np
from scipy.interpolate import CubicSpline

from .base import IndexedSet


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


# def find_cubic_coefficients(x1, y1, x2, y2):
#     # Set up the system of equations:
#     # a*x1^3 + b*x1^2 = y1
#     # a*x2^3 + b*x2^2 = y2
#     A = np.array([[x1**3, x1**2], [x2**3, x2**2]])
#     Y = np.array([y1, y2])

#     # Solve for a and b
#     a, b = np.linalg.solve(A, Y)
#     return a, b


# def classify_uscs(sizes, percent_passing):
#     """
#     Classify a grain size distribution according to USCS standards.

#     Parameters:
#     -----------
#     sizes : array-like
#         Array of sieve sizes in mm
#     percent_passing : array-like
#         Array of percent passing values corresponding to sizes

#     Returns:
#     --------
#     str
#         USCS classification symbol
#     """
#     sizes = np.asarray(sizes)
#     percent_passing = np.asarray(percent_passing)

#     # Key sieve sizes (mm)
#     sieve_200 = 0.075
#     sieve_4 = 4.75

#     # Interpolate percent passing at key sizes
#     percent_fines = np.interp(sieve_200, sizes, percent_passing)
#     percent_gravel = 1 - np.interp(sieve_4, sizes, percent_passing)

#     # Coarse-grained soils (< 50% passing #200)
#     if percent_fines < 0.5:
#         if percent_gravel > 0.5:
#             # Gravel
#             if percent_fines < 0.05:
#                 # Clean gravel - check gradation
#                 cu, cc = _gradation_parameters(sizes, percent_passing)
#                 if cu >= 4 and 1 <= cc <= 3:
#                     return "GW"  # Well-graded gravel
#                 else:
#                     return "GP"  # Poorly graded gravel
#             elif 0.05 <= percent_fines <= 0.12:
#                 # Borderline - dual symbol
#                 cu, cc = _gradation_parameters(sizes, percent_passing)
#                 if cu >= 4 and 1 <= cc <= 3:
#                     return "GW-GM" if _is_silty(sizes, percent_passing) else "GW-GC"
#                 else:
#                     return "GP-GM" if _is_silty(sizes, percent_passing) else "GP-GC"
#             else:
#                 # Gravel with fines
#                 return "GM" if _is_silty(sizes, percent_passing) else "GC"
#         else:
#             # Sand
#             if percent_fines < 0.05:
#                 # Clean sand - check gradation
#                 cu, cc = _gradation_parameters(sizes, percent_passing)
#                 if cu >= 6 and 1 <= cc <= 3:
#                     return "SW"  # Well-graded sand
#                 else:
#                     return "SP"  # Poorly graded sand
#             elif 0.05 <= percent_fines <= 0.12:
#                 # Borderline - dual symbol
#                 cu, cc = _gradation_parameters(sizes, percent_passing)
#                 if cu >= 6 and 1 <= cc <= 3:
#                     return "SW-SM" if _is_silty(sizes, percent_passing) else "SW-SC"
#                 else:
#                     return "SP-SM" if _is_silty(sizes, percent_passing) else "SP-SC"
#             else:
#                 # Sand with fines
#                 return "SM" if _is_silty(sizes, percent_passing) else "SC"

#     # Fine-grained soils (≥ 50% passing #200)
#     else:
#         # For fine-grained soils, Atterberg limits are needed for proper classification
#         # Return generic classifications
#         return (
#             "ML/CL"  # Silt/clay - requires Atterberg limits for precise classification
#         )


def _gradation_parameters(sizes, percent_passing):
    """
    Calculate uniformity coefficient (Cu) and coefficient of curvature (Cc).

    Parameters:
    -----------
    sizes : array-like
        Array of sieve sizes in mm
    percent_passing : array-like
        Array of percent passing values

    Returns:
    --------
    tuple
        (Cu, Cc) - uniformity coefficient and coefficient of curvature
    """
    # Find D10, D30, D60 (grain sizes at 10%, 30%, 60% passing)
    d10 = np.interp(0.10, percent_passing, sizes)
    d30 = np.interp(0.30, percent_passing, sizes)
    d60 = np.interp(0.60, percent_passing, sizes)

    cu = d60 / d10 if d10 > 0 else np.inf
    cc = (d30**2) / (d60 * d10) if d60 > 0 and d10 > 0 else 0

    return cu, cc


def _is_silty(sizes, percent_passing):
    """
    Determine if fine fraction is silty based on grain size distribution.

    This is a simplified classification - proper classification requires
    Atterberg limit tests.

    Parameters:
    -----------
    sizes : array-like
        Array of sieve sizes in mm
    percent_passing : array-like
        Array of percent passing values

    Returns:
    --------
    bool
        True if classified as silty, False if clayey
    """
    # This is a simplified heuristic - actual classification requires plasticity tests
    # For demonstration, assume silty if distribution is more uniform in fine range
    fine_sizes = sizes[sizes <= 0.075]
    if len(fine_sizes) > 1:
        fine_passing = percent_passing[sizes <= 0.075]
        # Simple heuristic: more uniform = silty
        return np.std(fine_passing) < 0.20
    return True  # Default to silty


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

            spline = CubicSpline(log_size, self.percent_passing, bc_type="natural")
            # extend the natural natural spline with linear extrapolating knots
            add_boundary_knots(spline)
            log_size = np.linspace(0.0001, log_size[-1], 1000)
            gsd_curve = spline(log_size, nu=0)

        log_d = np.interp(percent, log_size, gsd_curve)
        return 10**log_d

    def _i_gs_curve(self):
        """
        Extend the GSD curve over tha required range for defining the grain-size index, I_GS as described by (Erguler, 2016).
        The range is from 0.001 to 75 mm."""
        left_bound = 0.0001
        right_bound = 75.0

        size_points = self.sizes
        gsd_curve = self.percent_passing
        n_extend = 10

        # These commented out lines complete I_GS, but I'm turning them off for now.
        # This makes a modified I_GS that is more appropriate for the MPS work I'm doing right now.
        # if size_points[-1] < right_bound:
        #     size_points = np.concatenate(
        #         [size_points, np.linspace(size_points[-1], right_bound, n_extend)]
        #     )
        #     gsd_curve = np.concatenate([gsd_curve, np.ones(n_extend) * gsd_curve[-1]])

        log_size = np.log10(size_points)

        if size_points[0] > left_bound:
            spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
            add_boundary_knots(spline)
            log_size = np.linspace(np.log10(left_bound), log_size[-1], 100)
            gsd_curve = spline(log_size, nu=0)

        return log_size, gsd_curve

    @property
    def max_curvature(self):
        """
        Calculate the maximum curvature of the GSD curve.

        Returns:
        --------
        float
            Maximum curvature of the GSD curve
        """
        log_size, gsd_curve = self._i_gs_curve()
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        second_derivative = spline(log_size, nu=2)
        return np.max(np.abs(second_derivative))

    @property
    def max_positive_curvature(self):
        """
        Calculate the maximum positive curvature of the GSD curve.

        Returns:
        --------
        float
            Maximum positive curvature of the GSD curve
        """
        log_size, gsd_curve = self._i_gs_curve()
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        second_derivative = spline(log_size, nu=2)
        return (
            np.max(second_derivative[second_derivative > 0])
            if np.any(second_derivative > 0)
            else 0.0
        )

    @property
    def max_negative_curvature(self):
        """
        Calculate the maximum negative curvature of the GSD curve.

        Returns:
        --------
        float
            Maximum negative curvature of the GSD curve
        """
        log_size, gsd_curve = self._i_gs_curve()
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        second_derivative = spline(log_size, nu=2)
        return (
            np.min(second_derivative[second_derivative < 0])
            if np.any(second_derivative < 0)
            else 0.0
        )

    @property
    def concavity_on_linear(self):
        """
        Calculate the concavity of the GSD curve.

        Returns:
        --------
        float
            Concavity of the GSD curve, defined as the maximum curvature divided by the range of sizes
        """
        log_size, gsd_curve = self._i_gs_curve()
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        second_derivative = spline(10**log_size, nu=2)
        return second_derivative

    @property
    def concavity_on_log(self):
        """
        Calculate the concavity of the GSD curve.

        Returns:
        --------
        float
            Concavity of the GSD curve, defined as the maximum curvature divided by the range of sizes
        """
        log_size, gsd_curve = self._i_gs_curve()
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        second_derivative = spline(log_size, nu=2)
        return second_derivative

    @property
    def average_curvature(self):
        """
        Calculate the average curvature of the GSD curve.

        Returns:
        --------
        float
            Average curvature of the GSD curve, defined as the integral of the absolute value of the second derivative
            divided by the range of sizes
        """
        log_size, gsd_curve = self._i_gs_curve()
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        second_derivative = spline(log_size, nu=2)
        integral_curvature = np.trapz(np.abs(second_derivative), log_size)
        size_range = log_size[-1] - log_size[0]
        return integral_curvature / size_range if size_range > 0 else np.inf

    @property
    def max_slope(self):
        """
        Calculate the maximum slope of the GSD curve.

        Returns:
        --------
        float
            Maximum slope of the GSD curve
        """
        log_size, gsd_curve = self._i_gs_curve()
        spline = CubicSpline(log_size, gsd_curve, bc_type="natural")
        add_boundary_knots(spline)
        first_derivative = spline(log_size, nu=1)
        return np.max(np.abs(first_derivative))

    @property
    def i_gs(self):
        """
        Calculate the grain-size index (I_GS) as described by Erguler (2016).
        """
        log_size, gsd_curve = self._i_gs_curve()
        a_t = log_size[-1] - log_size[0]
        a_c = np.trapezoid(gsd_curve, log_size)
        return a_c / a_t

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

    def uscs_classification(self):
        """
        Classify this GSD according to USCS standards.

        Returns:
        --------
        str
            USCS classification symbol
        """
        return classify_uscs(self.sizes, self.percent_passing)
