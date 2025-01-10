import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def moving_average(data, window_size):
    """Applies a simple moving average to the data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def read_xyz(filename):
    with open(filename, 'r') as file:
        atoms = []
        n_atoms = int(file.readline().strip())
        file.readline()  # Skip the comment line
        for _ in range(n_atoms):
            line = file.readline().strip().split()
            x, y, z = float(line[1]), float(line[2]), float(line[3])
            atoms.append([x, y, z])
    return np.array(atoms)

def calculate_distances(atoms):
    n_atoms = len(atoms)
    distances = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(atoms[i] - atoms[j])
            distances.append(distance)
    return np.array(distances)

def compute_rdf(distances, dr, max_r, atoms):
    r = np.arange(0, max_r, dr)
    rdf = np.zeros(len(r) - 1)
    density = len(atoms) / (4/3 * np.pi * max_r**3)  # Number density for normalization
    for idx in range(len(r) - 1):
        shell_volume = 4/3 * np.pi * (r[idx+1]**3 - r[idx]**3)
        rdf[idx] = np.sum((distances >= r[idx]) & (distances < r[idx+1])) / (shell_volume * density)
    return r[:-1] + dr/2, rdf

# def plot_rdf_with_labels(r_values, rdf_values):
#     plt.figure(figsize=(10, 6))
#     plt.plot(r_values, rdf_values, label='RDF')
#     scatter_x_values = [3.3, 6, 7.5, 8.5, 10, 13, 15, 20, 30]
#     scatter_y_values = [rdf_values[np.abs(r_values - x).argmin()] for x in scatter_x_values]
#     plt.scatter(scatter_x_values, scatter_y_values, color='black', s=20, label='Selected Points', zorder=5)
#     text_offset_y = rdf_values.max() * 0.05  # Increased offset for text
#     for x, y in zip(scatter_x_values, scatter_y_values):
#         plt.text(x, y - text_offset_y, f'{x} Å', fontsize=8, ha='center', color='black', fontweight='bold')  # Reduced text size
#     plt.xlabel('Distance (angstroms)')
#     plt.ylabel('g(r)')
#     plt.title('Radial Distribution Function')
#     plt.grid(True)
#     plt.legend()
#     plt.ylim(bottom=None, top=rdf_values.max() * 1.2)  # Extends the y-axis limit to 120% of the highest value
#     plt.show()

#def plot_rdf_with_labels_v2(r_values, rdf_values):
    # Convert r_values from pixels to micrometers
    r_values_micrometers = r_values * 1.4
    scatter_x_values = [13.5, 22.5, 33, 42, 65]
    scatter_x_values_micrometers = [x * 1.4 for x in scatter_x_values]
    scatter_y_values = [rdf_values[np.abs(r_values - x).argmin()] for x in scatter_x_values]
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(r_values_micrometers, rdf_values, label='RDF', color='black', linewidth=0.5)
    ax.scatter(scatter_x_values_micrometers, scatter_y_values, color='red', s=30, zorder=5, facecolors='none', linewidths=1.5)
    
    text_offset_y = rdf_values.max() * 0.08  # Adjusted offset for text
    place_above = False

    # Uncomment and adjust the following lines if you want to add text labels to the scatter points
    # for x, y in zip(scatter_x_values_micrometers, scatter_y_values):
    #     if place_above:
    #         ax.text(x, y + text_offset_y, f'{x:.2f} µm', fontsize=6, ha='left', color='black')  # Place above
    #         place_above = False
    #     else:
    #         ax.text(x, y - text_offset_y, f'{x:.2f} µm', fontsize=6, ha='left', color='black')  # Place below
    #         place_above = True

    ax.set_xlabel('Distance (µm)', fontsize='medium')
    ax.set_ylabel('g(r)', fontsize='medium')
    ax.set_ylim(bottom=None, top=rdf_values.max() * 1.2)  # Extends the y-axis limit to 120% of the highest value
    ax.tick_params(axis='y', which='both', left=False, labelleft=False, labelsize='medium')
    ax.tick_params(axis='x', labelsize='medium')

    # Set x-ticks to be in micrometers
    x_ticks = np.arange(0, 65*1.4, 10)
    ax.set_xticks(x_ticks)
    ax.grid(False)
    plt.tight_layout()
    
    # Adjust the subplot to make sure the plot area has the desired dimensions
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.2)  # Adjust these values to control the plot area size
    
    plt.savefig("wave_RDF_v8.png")
    plt.close(fig)


from matplotlib import font_manager

# Ensure Nunito font is available
font_path = "Nunito-Bold.ttf"  # Update this path if necessary
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Nunito'
plt.rcParams['font.weight'] = 'bold'

def plot_rdf_with_labels_v2(r_values, rdf_values):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)  # Adjusted dimensions for a more square plot
    conv=1.4
    r_values=r_values*1.4
    ax.plot(r_values, rdf_values, label='RDF', color='black', linewidth=1.5)
    
    scatter_x_values = [13.5*conv, 22.5*conv, 33*conv, 42*conv, 65*conv,80*conv]
    scatter_y_values = [rdf_values[np.abs(r_values - x).argmin()] for x in scatter_x_values]
    ax.scatter(scatter_x_values, scatter_y_values, color='red', s=45, zorder=5, facecolors='none', linewidths=1.5)
    
    text_offset_y = rdf_values.max() * 0.08  # Adjusted offset for text
    place_above = False

    # Uncomment and adjust the following lines if you want to add text labels to the scatter points
    # for x, y in zip(scatter_x_values, scatter_y_values):
    #     if place_above:
    #         ax.text(x, y + text_offset_y, f'{x} Å', fontsize=6, ha='left', color='black')  # Place above
    #         place_above = False
    #     else:
    #         ax.text(x, y - text_offset_y, f'{x} Å', fontsize=6, ha='left', color='black')  # Place below
    #         place_above = True

    ax.set_xlabel('Distance (um)', fontsize=12, weight='bold')  # Added dot next to Å
    ax.set_ylabel('g(r)', fontsize=12, weight='bold')
    # Removed the legend
    ax.set_ylim(bottom=None, top=rdf_values.max() * 1.2)  # Extends the y-axis limit to 120% of the highest value
    ax.tick_params(axis='y', which='both', left=False, labelleft=False, labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(False)
    plt.tight_layout()
    
    # Adjust the subplot to make sure the plot area has the desired dimensions
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # Adjust these values to control the plot area size
    
    plt.savefig("Wave_RDF_v4.png")
    plt.close(fig)


def plot_rdf_with_labels_v3(r_values, rdf_values, window_size=5):
    # Apply moving average to smooth the RDF values
    smooth_rdf_values = moving_average(rdf_values, window_size)
    smooth_r_values = r_values[:len(smooth_rdf_values)]  # Adjust r_values to match the length of smoothed rdf_values

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(smooth_r_values, smooth_rdf_values, label='RDF', color='black', linewidth=0.5)
    scatter_x_values = [9.5,13,10.5, 22, 30, 40]
    scatter_y_values = [smooth_rdf_values[np.abs(smooth_r_values - x).argmin()] for x in scatter_x_values]

    ax.scatter(scatter_x_values, scatter_y_values, color='red', s=30, zorder=5,facecolors='none',linewidths=1.5)
    text_offset_y = rdf_values.max() * 0.08  # Adjusted offset for text
    place_above = False

    ax.set_xlabel('Distance (pixel)', fontsize='medium')
    ax.set_ylabel('g(r)', fontsize='medium')
    ax.set_ylim(bottom=None, top=rdf_values.max() * 1.2)  # Extends the y-axis limit to 120% of the highest value
    ax.tick_params(axis='y', which='both', left=False, labelleft=False, labelsize='medium')
    ax.tick_params(axis='x', labelsize='medium')
    x_ticks = range(0, 50, 5)  # Adjust range as needed
    ax.set_xticks(x_ticks)
    ax.grid(False)
    plt.tight_layout()
    
    # Adjust the subplot to make sure the plot area has the desired dimensions
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.2)  # Adjust these values to control the plot area size
    
    plt.savefig("wave_RDF_v7.png")
    plt.close(fig)

# Run the main function
filename = 'trajectory.xyz'  # Use the uploaded file path
atoms = read_xyz(filename)
distances = calculate_distances(atoms)
dr = 1  # Histogram bin width
max_r = 80.0  # Maximum distance considered
r_values, rdf_values = compute_rdf(distances, dr, max_r, atoms)

#plot_rdf_with_labels(r_values, rdf_values)
plot_rdf_with_labels_v2(r_values, rdf_values)
#plot_rdf_with_labels_v3(r_values, rdf_values, window_size=3)  # Adjust window_size for more or less smoothing




