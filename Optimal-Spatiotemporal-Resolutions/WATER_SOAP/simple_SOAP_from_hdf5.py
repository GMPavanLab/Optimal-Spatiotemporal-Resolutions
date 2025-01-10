# Import necessary libraries
import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
import dynsight
from dynsight._internal.hdf5er.from_hdf5 import create_universe_from_slice
from dynsight._internal.soapify.utilities import fill_soap_vector_from_dscribe, get_soap_settings
from dynsight._internal.utilities.utilities import normalize_array

# Define variables
NAME = 'WaterCOEX'                    # Name of your HDF5 file (without extension)
GROUP = "/Trajectories/" + NAME
HDF5_TRJ = NAME + ".hdf5"
SOAP_CUTOFF = 25                     # The cutoff radius for SOAP computation
pc_components = 1                     # Number of principal components to keep

# Ensure output directory exists
os.makedirs('arrays', exist_ok=True)

# Step 1: Create Universe from HDF5 file
print("Creating Universe from HDF5 file...")

try:
    with h5py.File(HDF5_TRJ, "r") as trajFile:
        tgroup = trajFile[GROUP]
        universe = create_universe_from_slice(tgroup)
except Exception as e:
    print(f"An error occurred while creating the Universe: {e}")
    exit(1)

# Step 2: Prepare the HDF5 file for SOAP data
print("Preparing HDF5 file for SOAP data...")

try:
    with h5py.File(HDF5_TRJ, "a") as file:
        if "SOAP" not in file:
            file.create_group("SOAP")
except Exception as e:
    print(f"An error occurred while preparing the HDF5 file for SOAP data: {e}")
    exit(1)

# Step 3: Compute SOAP descriptors
print(f"Computing SOAP descriptors with cutoff radius {SOAP_CUTOFF} Å...")
try:
    with h5py.File(HDF5_TRJ, "a") as file:
        soap_group = file["SOAP"].require_group(f"SOAP_{int(SOAP_CUTOFF)}")
        soap_dataset_name = 'WaterCOEX'
        if soap_dataset_name in soap_group:
            print(f"SOAP descriptors for cutoff radius {SOAP_CUTOFF} Å already exist. Skipping computation.")
        else:
            # Compute SOAP descriptors using dynsight
            dynsight.soapify.saponify_trajectory(
                trajcontainer=file[GROUP],
                soapoutcontainer=soap_group,
                verbose=False,
                soapnmax=8,
                soaplmax=8,
                soapoutputchunkdim=50,
                soapnjobs=8,
                soaprcut=SOAP_CUTOFF,
                soap_respectpbc=True
            )
except Exception as e:
    print(f"An error occurred during SOAP computation: {e}")
    exit(1)

# Step 4: Extract the SOAP descriptors from the HDF5 file
print("Extracting SOAP descriptors...")
try:
    with h5py.File(HDF5_TRJ, "r") as file:
        soap_group = file["SOAP"][f"SOAP_{int(SOAP_CUTOFF)}"]
        datasets = list(soap_group.keys())
        print("Datasets in SOAP group:", datasets)
        
        # Assuming the dataset containing SOAP vectors is named 'WaterCOEX'
        if 'WaterCOEX' in datasets:
            soap_dataset = soap_group['WaterCOEX']
            
            # Get settings for SOAP computation
            fillSettings = get_soap_settings(soap_dataset)
            fillSettings = {k.lower(): v for k, v in fillSettings.items()}  # Convert keys to lowercase
            
            # Extract the SOAP descriptors
            soap_data = soap_dataset[:]
            print(f"Extracted SOAP data shape: {soap_data.shape}")
            
            # Fill SOAP vector from dscribe
            soap_filled = fill_soap_vector_from_dscribe(soap_data, **fillSettings)
            print(f"SOAP data after filling from dscribe shape: {soap_filled.shape}")
            
            # Transpose the SOAP array to have atoms as the first dimension
            soap_array = soap_filled.transpose(1, 0, 2)  # Now shape is (n_atoms, n_frames, n_features)
            print(f"SOAP descriptors shape after transposing axes: {soap_array.shape}")
        else:
            raise KeyError("Dataset 'WaterCOEX' not found in the SOAP group.")
except Exception as e:
    print(f"An error occurred while extracting SOAP descriptors: {e}")
    exit(1)

# Step 5: Reshape the SOAP array for PCA
print(f"SOAP descriptors shape before reshaping: {soap_array.shape}")
try:
    # soap_array has shape (n_atoms, n_frames, n_features)
    n_atoms, n_frames, n_features = soap_array.shape
    print(f"Number of atoms: {n_atoms}")
    print(f"Number of frames: {n_frames}")
    print(f"Number of features: {n_features}")
    
    # Reshape to (n_atoms * n_frames, n_features)
    soap_reshaped = soap_array.reshape(n_atoms * n_frames, n_features)
    print(f"Reshaped SOAP array shape for PCA: {soap_reshaped.shape}")
except Exception as e:
    print(f"An error occurred while reshaping the SOAP array: {e}")
    exit(1)

# Step 6: Perform PCA
print("Performing PCA...")
try:
    pca = PCA(n_components=pc_components)
    pc_soap = pca.fit_transform(soap_reshaped)
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance ratio:", explained_variance)
    print(f"PCA result shape: {pc_soap.shape}")
except Exception as e:
    print(f"An error occurred during PCA: {e}")
    exit(1)

# Step 7: Save the PCA-transformed data as an NPZ file
print("Saving the PCA-transformed data...")
try:
    # Reshape back to (n_atoms, n_frames, pc_components)
    pc_soap_reshaped = pc_soap.reshape(n_atoms, n_frames, pc_components)
    print(f"PCA-transformed data reshaped to: {pc_soap_reshaped.shape}")
    
    output_filename = f"arrays/SOAP_PCA_{pc_components}_components.npz"
    np.savez_compressed(output_filename, PCA_Soap=pc_soap_reshaped)
    print(f"PCA-transformed data saved to {output_filename}")
except Exception as e:
    print(f"An error occurred while saving the PCA-transformed data: {e}")
    exit(1)

