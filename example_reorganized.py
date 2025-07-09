"""
Example script demonstrating the reorganized gsd_lib package.
"""

from gsd_lib import GSD, MinimalPackingGenerator

if __name__ == "__main__":
    # Example usage similar to the original main function
    d = [9.96705158, 26.63032713, 32.60142395, 33.56999312, 51.61020638]
    mass = [6.93546639, 12.49031735, 9.29866496, 8.61983351, 0.0]

    # Create GSD
    g = GSD(sizes=d, masses=mass)

    # Create minimal packing generator
    s = MinimalPackingGenerator(g)

    # Print results
    print(f"Errors: {g.description_error(s.mps)}")
    print(f"Sample quantities: {s.mps.quantities}")
