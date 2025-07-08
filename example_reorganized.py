"""
Example script demonstrating the reorganized gsd_lib package.
"""

from gsd_lib import GSD, MinimalPackingGenerator

if __name__ == "__main__":
    # Example usage similar to the original main function
    r = [0.111, 0.222, 0.333, 0.444, 0.555]
    weight = [0.39, 0.20, 0.14, 0.27, 0.0]

    # Create GSD
    g = GSD(sizes=r, masses=weight)

    # Create minimal packing generator
    s = MinimalPackingGenerator(g, max_size="representative", order=2, tol=0.000001)

    # Print results
    print(f"Errors: {g.description_error(s.mps)}")
    print(f"Sample quantities: {s.mps.quantities}")
