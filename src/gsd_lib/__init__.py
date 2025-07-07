"""
GSD-Lib: Discrete analytical mechanics library for grain size distributions.

This package provides classes for working with grain size distributions
and indexed sets of particles.
"""

from .core import IndexedSet, GSD, gsd_gen, sphere_vol, sphere_rad

__version__ = "0.1.0"
__all__ = ["IndexedSet", "GSD", "gsd_gen", "sphere_vol", "sphere_rad"]