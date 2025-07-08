"""GSD (Grain Size Distribution) class."""

import warnings

import numpy as np

from .base import IndexedSet


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
                (sample.sizes >= self.sizes[i]) & (sample.sizes < self.sizes[i + 1])
            )[0]  # TODO: fix greater than or equal to
            percent_between = np.sum(sample.total_masses[sample_indices]) / sample_mass
            sample_percent_retained.append(percent_between)

        retained_on_last = (
            np.sum(sample.total_masses[sample.sizes > self.sizes[-1]]) / sample_mass
            if len(sample.sizes) > 0
            else 0.0
        )
        sample_percent_retained.append(retained_on_last)

        return np.array(sample_percent_retained) - self.percent_retained
