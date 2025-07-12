"""USCS (Unified Soil Classification System) classification functions."""

import numpy as np


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
