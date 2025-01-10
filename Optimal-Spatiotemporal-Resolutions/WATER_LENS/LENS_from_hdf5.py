import argparse
import h5py
import numpy as np
import dynsight
import os

# Initialize parser
parser = argparse.ArgumentParser(description='Process trajectory data.')

# Adding arguments
parser.add_argument('-f', '--filename', type=str, required=True, help='Path to the trajectory HDF5 file')
parser.add_argument('-a', '--address', type=str, required=True, help='Address within the HDF5 file to the trajectory data')
parser.add_argument('-c', '--cutoff', type=float, default=10, help='Cutoff distance for analysis')

# Read arguments from command line
args = parser.parse_args()

trajFileName = args.filename
trajAddress = args.address
cutoff = args.cutoff

# Check if the file exists
if not os.path.isfile(trajFileName):
    print(f"Error: The file {trajFileName} does not exist.")
    exit(1)

wantedTrajectory = slice(0, None, 1)

try:
    with h5py.File(trajFileName, "r") as trajFile:
        if trajAddress not in trajFile:
            print(f"Error: The address {trajAddress} was not found in the file.")
            exit(1)
        tgroup = trajFile[trajAddress]
        print("Creating universe from the trajectory slice...")
        universe = dynsight.hdf5er.create_universe_from_slice(tgroup, wantedTrajectory)
except Exception as e:
    print(f"An error occurred while processing the HDF5 file: {e}")
    exit(1)

try:
    nAtoms = len(universe.atoms)
    print(f"Number of atoms: {nAtoms}")
    print(f"Trajectory shape: {np.shape(universe.trajectory)}")

    print("Listing neighbours along the trajectory...")
    neigcounts = dynsight.lens.list_neighbours_along_trajectory(input_universe=universe,cutoff=cutoff)
    print("Calculating neighbour change over time...")
    LENS, nn, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)
    np.savez("name.npz", LENS)
    print("Analysis completed and data saved.")
except Exception as e:
    print(f"An error occurred during analysis: {e}")
    exit(1)
