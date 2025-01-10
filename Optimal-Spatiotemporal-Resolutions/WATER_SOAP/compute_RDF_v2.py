# ruff: noqa: F403, F405
import matplotlib
matplotlib.use('Agg')  # Ensure matplotlib runs in headless mode.
from ovito.io import import_file
from ovito.modifiers import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
import sys

def main():
    try:
        print("Loading file...")
        pipeline = import_file("./LENS_WaterCOEX.xyz", columns=["Particle Type", "Position.X", "Position.Y", "Position.Z"])
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    try:
        print("Applying ExpressionSelectionModifier...")
        modifier = ExpressionSelectionModifier(expression="ParticleType >= 0")
        pipeline.modifiers.append(modifier)
    except Exception as e:
        print(f"Error applying ExpressionSelectionModifier: {e}")

    try:
        print("Applying CoordinationAnalysisModifier...")
        modifier = CoordinationAnalysisModifier(cutoff=30.0, number_of_bins=100)
        pipeline.modifiers.append(modifier)
    except Exception as e:
        print(f"Error applying CoordinationAnalysisModifier: {e}")

    try:
        print("Applying TimeAveragingModifier...")
        modifier = TimeAveragingModifier(operate_on='table:coordination-rdf')
        pipeline.modifiers.append(modifier)
    except Exception as e:
        print(f"Error applying TimeAveragingModifier: {e}")

    try:
        print("Computing pipeline and retrieving RDF data...")
        total_rdf = pipeline.compute().tables['coordination-rdf[average]'].xy()
        rdf_bins = total_rdf[:, 0]
        rdf = total_rdf[:, 1]
        
        # Printing the rdf_bins and rdf arrays
        print("rdf_bins:", rdf_bins)
        print("rdf:", rdf)
    except KeyError as e:
        print(f"KeyError while accessing RDF data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during pipeline computation or RDF retrieval: {e}")
        sys.exit(1)

    try:
        print("Finding minima in RDF data...")
        minima_indices = argrelextrema(rdf, np.less)
        # Print minima_indices
        print("minima_indices:", minima_indices)
    except Exception as e:
        print(f"Error finding minima: {e}")

    try:
        # Print size and shape of arrays to check for any issues
        print(f"rdf_bins shape: {rdf_bins.shape}, rdf shape: {rdf.shape}")
        print(f"Minima indices shape: {minima_indices[0].shape if minima_indices else 'None'}")
        
        # Plot RDF data and minima points
        print("Plotting RDF data and minima points...")
        plt.plot(rdf_bins, rdf, label="Array Values", color="black")
        plt.scatter(rdf_bins[minima_indices[0]], rdf[minima_indices], color='red', label="Minima Points")
        plt.title("Array and Minima Points")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig("RDF.png")
        print("Plot saved as RDF.png")
    except Exception as e:
        print(f"Error during plotting or saving figure: {e}")

if __name__ == "__main__":
    main()

