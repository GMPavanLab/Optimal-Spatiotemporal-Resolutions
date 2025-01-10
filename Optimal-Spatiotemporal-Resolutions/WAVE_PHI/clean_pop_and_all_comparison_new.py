
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import glob
from scipy.interpolate import make_interp_spline
import re
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib import font_manager
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline
#Onion plot emulator

from matplotlib import font_manager, font_manager as fm
# Ensure Nunito font is available
font_path = "Nunito-Bold.ttf"  # Update this path if necessary
font_manager.fontManager.addfont(font_path)
nunito_font = fm.FontProperties(fname=font_path)


plt.rcParams['font.family'] = 'Nunito'
#plt.rcParams['font.weight'] = 'bold'



def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

def plot_onion_data_v13(all_data):
    # Extract onion labels and tau windows for the plot
    onion_labels = []
    tau_windows = set()
    tau_indices = {}

    # Extract the numeric part of onion folder labels and collect all tau_windows
    for win, mean_env0, n_states, onion_label in all_data:
        match = re.search(r"ONION_(\d+\.?\d*)", onion_label)
        if match:
            onion_num = float(match.group(1))
            onion_labels.append((onion_num, onion_label))
            tau_windows.update(tau for tau, env in zip(win, mean_env0) if env < 0.5)  # Consider tau_windows with env0 < 0.5

    onion_labels.sort()
    tau_windows = sorted(tau_windows)

    print("label onion 1:",onion_labels)
    # Map each tau_window to its index
    for idx, tau in enumerate(tau_windows):
        tau_indices[tau] = idx

    # Generate plot data
    mean_n_states_per_cutoff = {onion_num: [] for onion_num, _ in onion_labels}
    for onion_num, onion_label in onion_labels:
        for win, mean_env0, n_states, label in all_data:
            if label == onion_label:
                # Filter time windows > 3 and env0 > 0.5
                filtered_n_states = [n_states[idx] for idx, (tau, env) in enumerate(zip(win, mean_env0)) if tau > 0 and env < 0.5]
                if filtered_n_states:
                    mean_n_states_per_cutoff[onion_num].append(np.mean(filtered_n_states))
                break

    # Prepare data for plotting
    mean_n_states_values = [np.mean(mean_n_states_per_cutoff[onion_num]) for onion_num, _ in onion_labels]

    # Remove the points with x = 7.5, 11.5, and 12.5 if they exist
    filtered_labels = [(onion_num, onion_label) for onion_num, onion_label in onion_labels if onion_num not in [3.4,7.5, 11.4, 12.4]]
    filtered_mean_n_states_values = [mean_n_states_values[i] for i, (onion_num, _) in enumerate(onion_labels) if onion_num not in [3.4,7.5, 11.4, 12.4]]
    print("label onion:",onion_labels)
    print("label filtered:",filtered_labels)
    # Create the plot
    #sns.set(style="white")
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=300)

    # X-axis labels and positions
    x_labels = [onion_num for onion_num, _ in filtered_labels]  # Extract numeric part for x-axis labels
    
    # Scatter plot of the mean_n_states_values
    ax1.scatter(x_labels, filtered_mean_n_states_values, color='green', edgecolor='black')

    p1 = np.poly1d(np.polyfit(x_labels, filtered_mean_n_states_values, 4))
    x_new_1 = np.linspace(min(x_labels), max(x_labels), 300)
    y_smooth_1 = p1(x_new_1)

    ax1.plot(x_new_1, y_smooth_1, color='green', linestyle='-', linewidth=3)
    

    # Customize the primary y-axis
    ax1.set_xlabel('Descriptor CutOff (Å)', fontsize=12,fontproperties=nunito_font)
    ax1.set_ylabel('Mean # of ENVs', fontsize=12, color='green',fontproperties=nunito_font)

    # Customize the tick parameters to use Nunito font
    #ax1.tick_params(axis='x', labelsize=12, labelcolor='black', width=2, length=6, direction='inout')
    ax1.tick_params(axis='y', labelsize=12, labelcolor='green')
    # ax1.tick_params(which='both', width=2, length=6)
    # ax1.xaxis.set_tick_params(which='both', width=2, length=6)
    # ax1.yaxis.set_tick_params(which='both', width=2, length=6)

    for label in ax1.get_xticklabels():
        label.set_fontproperties(nunito_font)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(nunito_font)
        #label.set_fontweight('bold')
        label.set_color('green')
    
    # Customize the plot axes colors
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('green')
    ax1.spines['right'].set_color('black')
    ax1.set_xticks(range(int(min(x_labels)), int(max(x_labels)) + 1))
    ax1.xaxis.set_major_locator(plt.MultipleLocator(5))  # This ensures that every unit is ticked

    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Set spine width for consistency
    ax1.spines['left'].set_linewidth(2.5)
    ax1.set_ylim(1, max(filtered_mean_n_states_values) + 1)
    # Adjust layout to ensure x-axis labels are readable
    plt.tight_layout()
    
    # Adjust the subplot to make sure the plot area has the desired dimensions
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    # Save and show the plot
    plt.savefig('Onion_Data_Plot_final_v4.png')
    #plt.show()
    plt.close


    # Convert x_labels and y_values to numpy arrays
    x_data = np.array(x_labels, dtype=float)
    y_data = np.array(filtered_mean_n_states_values, dtype=float)

    # Sort the data (spline fitting may require sorted data)
    sorted_indices = np.argsort(x_data)
    x_data_sorted = x_data[sorted_indices]
    y_data_sorted = y_data[sorted_indices]

    # Fit a smoothing spline
    # The 's' parameter controls the amount of smoothing
    # Setting 's' to a positive value allows the spline to not pass through all data points
    smoothing_factor = len(x_data)  # Adjust this value as needed
    spline = UnivariateSpline(x_data_sorted, y_data_sorted, s=0.01)

    # Generate x values for the fitted curve
    x_spline = np.linspace(min(x_data_sorted), max(x_data_sorted), 300)
    y_spline = spline(x_spline)

    # Create a new figure
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=300)

    # Scatter plot of the original data
    ax1.scatter(x_data, y_data, color='green', edgecolor='black')

    # Plot the smoothing spline
    ax1.plot(x_spline, y_spline, color='green', linestyle='-', linewidth=3)

    # Customize the primary y-axis
    ax1.set_xlabel('Descriptor CutOff (Å)', fontsize=12, fontproperties=nunito_font)
    ax1.set_ylabel('Mean # of ENVs', fontsize=12, color='green', fontproperties=nunito_font)

    # Customize tick parameters and labels
    ax1.tick_params(axis='y', labelsize=12, labelcolor='green')
    for label in ax1.get_xticklabels():
        label.set_fontproperties(nunito_font)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(nunito_font)
        label.set_color('green')

    # Customize the plot axes colors and spines
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('green')
    ax1.spines['right'].set_color('black')
    ax1.spines['left'].set_linewidth(2.5)

    # Set x-axis ticks and locators
    ax1.set_xticks(range(int(min(x_data)), int(max(x_data)) + 1))
    ax1.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Set y-axis limits
    ax1.set_ylim(1, max(y_data) + 1)

    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    # Save and show the new plot
    plt.savefig('Onion_Data_Plot_SmoothingSpline.png')
    #plt.show()
    plt.close()


    # Convert x_labels and y_values to numpy arrays
    x_data = np.array(x_labels, dtype=float)
    y_data = np.array(filtered_mean_n_states_values, dtype=float)

    # Sort the data (if needed)
    sorted_indices = np.argsort(x_data)
    x_data_sorted = x_data[sorted_indices]
    y_data_sorted = y_data[sorted_indices]

    # Find the x corresponding to the maximum y
    max_y_index = np.argmax(y_data_sorted)
    x_max_y = x_data_sorted[max_y_index]
    y_max = y_data_sorted[max_y_index]
    y_min = min(y_data_sorted)

    # Define the shifted Gaussian function
    def shifted_gaussian(x, a, x0, sigma, y_min):
        return y_min + a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    # Set parameters for the Gaussian curve
    a = y_max - y_min  # Adjusted amplitude so that the peak reaches y_max
    x0 = x_max_y+1.7       # Center of the Gaussian (x-value of maximum y)
    sigma = (max(x_data_sorted) - min(x_data_sorted)) / 6  # Width of the Gaussian

    # Generate x values for plotting the Gaussian curve
    x_gauss = np.linspace(min(x_data_sorted)-1, max(x_data_sorted)+1, 300)
    y_gauss = shifted_gaussian(x_gauss, a, x0, 2/3*sigma, y_min)

    # Create a new figure
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=300)
    	
    # Scatter plot of the original data
    ax1.scatter(x_data, y_data, color='white', edgecolor='green', marker='o',linewidth=2)
    i_max = np.argmax(y_data)
    ax1.scatter(x_data[i_max], y_data[i_max], color='green', edgecolor='green', marker='*', s=250)
    # Plot the Gaussian curve with reduced line width
    ax1.plot(x_gauss, y_gauss, color='green', linestyle='-', linewidth=1.5)

    # Customize the primary y-axis
    ax1.set_xlabel('Descriptor CutOff (Å)', fontsize=12, fontproperties=nunito_font)
    ax1.set_ylabel('Mean # of ENVs', fontsize=12, color='green', fontproperties=nunito_font)

    # Customize tick parameters and labels
    ax1.tick_params(axis='y', labelsize=12, labelcolor='green')
    for label in ax1.get_xticklabels():
        label.set_fontproperties(nunito_font)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(nunito_font)
        label.set_color('green')

    # Customize the plot axes colors and spines
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_color('black')
    ax1.spines['left'].set_color('green')
    ax1.spines['right'].set_color('black')
    ax1.spines['left'].set_linewidth(2.5)

    # Set x-axis ticks and locators
    ax1.set_xticks(range(int(min(x_data)), int(max(x_data)) + 1))
    ax1.xaxis.set_major_locator(plt.MultipleLocator(5))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Set y-axis limits using the minimum and maximum y-values
    ax1.set_ylim(y_min - 1, y_max + 1)

    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    # Save the new plot
    plt.savefig('Onion_Data_Plot_GaussianCurve_Adjusted_v2.png')
    # plt.show()
    plt.close()



onion_folder = "."
threshold = 0.00012


onion_folders = glob.glob("ONION_*")
all_data = []
if not onion_folders:
    print("No ONION_* folders found.")
for onion in onion_folders:
        
    #collect data from folders

    folder = f"{onion}"
    
    # Reading files
    tau_window = []
    fraction_env0 = []
    with open(f"{folder}/fraction_0.txt", 'r') as file:
        for line in file:
            if(line.startswith("#")):
                continue
            values = line.split()  
            if values:
                tau_window.append(float(values[0]))  
                fraction_env0.append(float(values[1]))
    
    states_pop = {i: [] for i in tau_window}
    with open(f"{folder}/final_states.txt", 'r') as file:
        a = -1
        for line in file:
            if(line.startswith("##")):
                a += 1
                continue
            if(line.startswith("# ")):
                continue
            values = line.split()
            if values:
                try:
                    states_pop[tau_window[a]].append(float(values[3]))  
                except IndexError:
                    break
    n_states = []
    with open(f"{folder}/number_of_states.txt", 'r') as file:
        for line in file:
            if(line.startswith("#")):
                continue
            values = line.split()
            if values:
                n_states.append(int(values[1]))
    print("tau_window:", tau_window)
    print("fraction in ENV0:", fraction_env0)
    print("number of states:", n_states)
    print("states pop:", states_pop)
    print("---")
    a = 0
    clean_states_pop = {i: [] for i in tau_window}
    for key, value in states_pop.items():
        for i in range(len(value)):
            if(value[i] < threshold):
                print(f"win: {key}: LOW POPULATION DETECTED")
                fraction_env0[a] += value[i]
                n_states[a] -= 1
                continue
            clean_states_pop[key].append(value[i])
        a += 1



    # Debug
    print("tau_window:", tau_window)
    print("fraction in ENV0:", fraction_env0)
    print("number of states:", n_states)
    print("states pop:", states_pop)
    print("clean states pop:", clean_states_pop)
    # 
    tau_window = np.array(tau_window)
    fraction_env0 = np.array(fraction_env0)
    n_states = np.array(n_states)   

    
    
    all_data.append((tau_window, fraction_env0, n_states, onion))
    #print(f"Data from {sigma_folder}: win={win}, mean_env0={mean_env0}, mode_sts={mode_sts}")

if all_data:
    plot_onion_data_v13(all_data)
else:
    print("No data to plot.")


# 
