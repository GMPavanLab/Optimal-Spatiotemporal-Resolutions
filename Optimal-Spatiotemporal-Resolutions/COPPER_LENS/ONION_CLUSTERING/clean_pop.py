import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import font_manager


# Ensure Nunito font is available
font_path = "../Nunito-Bold.ttf"  # Update this path if necessary
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Nunito'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'  # Ensure that math text uses the regular font family

def onion_plot_v2(win, env0, states, log):
    win = win *0.1
    fig, axes = plt.subplots(figsize=(5, 5), dpi=300)
    
    # Plotting the data on the left axis
    axes.plot(win, states, marker="o", c="#1f77b4")
    axes.set_xlabel(r"Time resolution $\Delta t$ " + "[ns]", fontsize=12)
    axes.set_ylabel(r"# of ENVs", weight="bold", c="#1f77b4", fontsize=12)
    
    # Ensuring all left axis components are blue
    axes.yaxis.label.set_color("#1f77b4")
    axes.tick_params(axis='y', colors="#1f77b4", labelsize=12)
    for label in axes.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color("#1f77b4")
        label.set_fontsize(12)
    for tick in axes.yaxis.get_major_ticks():
        tick.label1.set_color('#1f77b4')
        tick.label1.set_fontweight('bold')
        tick.label1.set_fontsize(12)
    axes.tick_params(axis='y', colors="#1f77b4")

    axes.tick_params(axis='x', labelsize=12)

    if log:
        axes.set_xscale("log")
        axes.set_xlim(win[0] * 0.75, win[-1] * 1.5)

    axes.yaxis.set_major_locator(MaxNLocator(integer=True))

    axes2 = axes.twiny()
    axes2.set_xlabel(r"Time resolution $\Delta t$ [frames]", fontsize=12)
    axes2.tick_params(axis='x', labelsize=12)

    if log:
        axes2.set_xscale("log")
        axes2.set_xlim(win[0] * 0.75/0.1 , win[-1] * 1.5/0.1)

    # Adding second y-axis on the right
    axesr = axes.twinx()
    axesr.plot(win, env0, marker="o", c="#ff7f0e")
    axesr.set_ylim(0.0, 1)
    axesr.set_ylabel("ENV$_{0}$ fraction", weight="bold", c="#ff7f0e", fontsize=12)
    
    axesr.yaxis.label.set_color("#ff7f0e")
    axesr.tick_params(axis='y', colors="#ff7f0e", labelsize=12)
    for label in axesr.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color("#ff7f0e")
        label.set_fontsize(12)
    for tick in axesr.yaxis.get_major_ticks():
        tick.label2.set_color('#ff7f0e')
        tick.label2.set_fontweight('bold')
        tick.label2.set_fontsize(12)
    
    axes.set_ylim(-0.2, 4.1)
    axesr.spines['left'].set_color("#1f77b4")
    axesr.spines['right'].set_color("#ff7f0e")

    axesr.spines['right'].set_linewidth(2.5)
    axesr.spines['left'].set_linewidth(2.5)
    
    # Manually adjust the subplot parameters to ensure nothing is cut off and maintain a square plot area
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # Adjust these values to control the plot area size
    
    fig.savefig("Time_res_clean_final_v2.png")
    plt.close(fig)

onion_folder = "."
#adjust if necessary
threshold = 0.00012

# Reading files
tau_window = []
fraction_env0 = []
with open(f"{onion_folder}/fraction_0.txt", 'r') as file:
    for line in file:
        if(line.startswith("#")):
            continue
        values = line.split()  
        if values:
            tau_window.append(float(values[0]))  
            fraction_env0.append(float(values[1]))
 
states_pop = {i: [] for i in tau_window}
with open(f"{onion_folder}/final_states.txt", 'r') as file:
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
with open(f"{onion_folder}/number_of_states.txt", 'r') as file:
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

#tau_window = np.array(tau_window[:-1])
#fraction_env0 = np.array(fraction_env0[:-1])
#n_states = np.array(n_states[:-1])

#weird window to remove
#tau_window = np.delete(tau_window, 6)
#fraction_env0 = np.delete(fraction_env0, 6)
#n_states = np.delete(n_states, 6)


tau_window = np.array(tau_window)
fraction_env0 = np.array(fraction_env0)
n_states = np.array(n_states)
onion_plot_v2(tau_window,fraction_env0,n_states,True)
# 
