"""Utility functions for gsd_lib package."""

import numpy as np


def check_weights(contents):
    """Check that weights sum to 1."""
    if sum(contents) != 1:
        raise ValueError("weights must sum to 1")


def check_particles(contents):
    """Check that all particle values are positive."""
    if not all([x > 0 for x in contents]):
        raise ValueError("particles must be positive numbers")


def is_sorted(lst):
    """Check if a list is sorted in ascending order."""
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def sphere_vol(d):
    """Calculate volume based on diameter."""
    return (np.pi / 6) * d**3


def sphere_rad(v):
    """Calculate radius from volume."""
    return (3 * v / (4 * np.pi)) ** (1 / 3)


def sphere_diam(v):
    """Calculate diameter from volume."""
    return sphere_rad(v) * 2


def rep_size_by_vol(d_low, d_high):
    """Calculate representative size by volume between two sizes."""
    p = (2 - 4 ** (1 / 3)) * (d_low / d_high) + 4 ** (1 / 3)
    d_rep = d_low + (d_high - d_low) / p
    return d_rep
