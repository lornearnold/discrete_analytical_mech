"""
Example script demonstrating the reorganized gsd_lib package.
"""

import numpy as np

from gsd_lib import GSD, MinimalPackingGenerator

if __name__ == "__main__":
    # Example usage similar to the original main function
    d = np.array(
        [
            0.00075,
            0.075,
            0.15,
            0.3,
            0.6,
            1.18,
            2.36,
            4.75,
            9.5,
            19,
            25,
            37.5,
            50,
            63,
            75,
        ]
    )
    mass = np.array(
        [
            25.2964532,
            68.29616095,
            66.768287,
            7.42672594,
            95.60036628,
            16.4992489,
            66.98101603,
            80.85467591,
            46.77657895,
            62.97325543,
            19.05748664,
            19.52660537,
            78.82162845,
            8.83243329,
            0.0,
        ]
    )

    # Create GSD
    g = GSD(sizes=d, masses=mass)
    print(g.uscs_classification())

    # Create minimal packing generator
    s = MinimalPackingGenerator(g)

    # Print results
    print(f"Errors: {g.description_error(s.mps)}")
    print(f"Sample quantities: {s.mps.quantities}")
