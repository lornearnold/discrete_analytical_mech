"""GSD (Grain Size Distribution) class."""

import warnings

import numpy as np

from .base import IndexedSet


def classify_uscs(sizes, percent_passing):
    """
    Classify a grain size distribution according to USCS standards.

    Parameters:
    -----------
    sizes : array-like
        Array of sieve sizes in mm
    percent_passing : array-like
        Array of percent passing values corresponding to sizes

    Returns:
    --------
    str
        USCS classification symbol
    """
    sizes = np.asarray(sizes)
    percent_passing = np.asarray(percent_passing)

    # Key sieve sizes (mm)
    sieve_200 = 0.075
    sieve_4 = 4.75

    # Interpolate percent passing at key sizes
    percent_fines = np.interp(sieve_200, sizes, percent_passing)
    percent_gravel = 1 - np.interp(sieve_4, sizes, percent_passing)

    # Coarse-grained soils (< 50% passing #200)
    if percent_fines < 0.5:
        if percent_gravel > 0.5:
            # Gravel
            if percent_fines < 0.05:
                # Clean gravel - check gradation
                cu, cc = _gradation_parameters(sizes, percent_passing)
                if cu >= 4 and 1 <= cc <= 3:
                    return "GW"  # Well-graded gravel
                else:
                    return "GP"  # Poorly graded gravel
            elif 0.05 <= percent_fines <= 0.12:
                # Borderline - dual symbol
                cu, cc = _gradation_parameters(sizes, percent_passing)
                if cu >= 4 and 1 <= cc <= 3:
                    return "GW-GM" if _is_silty(sizes, percent_passing) else "GW-GC"
                else:
                    return "GP-GM" if _is_silty(sizes, percent_passing) else "GP-GC"
            else:
                # Gravel with fines
                return "GM" if _is_silty(sizes, percent_passing) else "GC"
        else:
            # Sand
            if percent_fines < 0.05:
                # Clean sand - check gradation
                cu, cc = _gradation_parameters(sizes, percent_passing)
                if cu >= 6 and 1 <= cc <= 3:
                    return "SW"  # Well-graded sand
                else:
                    return "SP"  # Poorly graded sand
            elif 0.05 <= percent_fines <= 0.12:
                # Borderline - dual symbol
                cu, cc = _gradation_parameters(sizes, percent_passing)
                if cu >= 6 and 1 <= cc <= 3:
                    return "SW-SM" if _is_silty(sizes, percent_passing) else "SW-SC"
                else:
                    return "SP-SM" if _is_silty(sizes, percent_passing) else "SP-SC"
            else:
                # Sand with fines
                return "SM" if _is_silty(sizes, percent_passing) else "SC"

    # Fine-grained soils (≥ 50% passing #200)
    else:
        # For fine-grained soils, Atterberg limits are needed for proper classification
        # Return generic classifications
        return (
            "ML/CL"  # Silt/clay - requires Atterberg limits for precise classification
        )


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

    @property
    def percent_passing(self):
        """
        Calculate percent passing for each size.

        Returns:
        --------
        np.ndarray
            Array of percent passing values corresponding to sizes
        """
        pp_offset = np.cumsum(self.percent_retained)
        pp = np.zeros(len(self.sizes))
        pp[1:] = pp_offset[:-1]
        return pp

    def uscs_classification(self):
        """
        Classify this GSD according to USCS standards.

        Returns:
        --------
        str
            USCS classification symbol
        """
        return classify_uscs(self.sizes, self.percent_passing)
