"""
GSD-Lib: A library for grain size distribution analysis and minimal packing generation.

This package provides tools for:
- Working with indexed sets of particle sizes
- Representing particle samples with quantities and properties
- Analyzing grain size distributions (GSD)
- Generating minimal packing configurations
"""

from .base import IndexedSet
from .generator import MinimalPackingGenerator
from .gsd import GSD
from .sample import Sample
from .utils import (
    check_particles,
    check_weights,
    is_sorted,
    rep_size_by_vol,
    sphere_diam,
    sphere_rad,
    sphere_vol,
)

__version__ = "0.1.0"

__all__ = [
    "IndexedSet",
    "Sample",
    "GSD",
    "MinimalPackingGenerator",
    "check_weights",
    "check_particles",
    "is_sorted",
    "sphere_vol",
    "sphere_rad",
    "sphere_diam",
    "rep_size_by_vol",
]
__all__ = [
    "IndexedSet",
    "Sample",
    "GSD",
    "MinimalPackingGenerator",
    "check_weights",
    "check_particles",
    "is_sorted",
    "sphere_vol",
    "sphere_rad",
    "sphere_diam",
    "rep_size_by_vol",
]
