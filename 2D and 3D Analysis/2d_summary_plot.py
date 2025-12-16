
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import chi2
from collections import defaultdict

plt.rcParams.update({
    "font.size": 20,        
    "axes.titlesize": 17,   
    "axes.labelsize": 17,   
    "xtick.labelsize": 17,  
    "ytick.labelsize": 17,  
    "legend.fontsize": 15   
})

#  CONFIGURATION 
DATA_DIRECTORY = "experimental_data_timed_Niconew"
PLOTS_DIRECTORY = "2d_summary_plots"
INTERPOLATION_POINTS = 101

#  SYSTEM GEOMETRY 
P_SX = np.array([0.0, 0.0])
P_DX = np.array([7.00, 0.0])
P_VTC = np.array([3.5, 24.5])
TRIANGLE_POINTS = np.array([P_SX, P_DX, P_VTC, P_SX])

def plot_2d_summary(list_of_dfs: list, subject_id: str, task_name: str, output_dir: str):
    """
    Analyzes a group of trials and generates a summary 2D plot.
    ALL ANALYSIS CODE MUST BE INSIDE THIS FUNCTION.
    """
    all_cop_points = []
    resampled_trajectories = []

    #  STEP 1: Process and normalize all repetitions 
    for df in list_of_dfs:
        df.replace('nan', np.nan, inplace=True)
        df.dropna(subset=['CoP_X', 'CoP_Y'], inplace=True)
        df = df.astype({'CoP_X': float, 'CoP_Y': float, 'F_tot': float})
        df = df[df['F_tot'] > 1.0].copy()
        if len(df) < 20: continue
        
        all_cop_points.append(df[['CoP_X', 'CoP_Y']].values)

        df['Time_s'] = (df['Timestamp'] - df['Timestamp'].iloc[0]) / 1000.0
        t_new = np.linspace(df['Time_s'].iloc[0], df['Time_s'].iloc[-1], INTERPOLATION_POINTS)
        resampled_trajectories.append(np.vstack((
            np.interp(t_new, df['Time_s'], df['CoP_X']),
            np.interp(t_new, df['Time_s'], df['CoP_Y']),
        )).T)

    if not all_cop_points:
        print(f"Skipping {subject_id}_{task_name}: insufficient data.")
        return # Exit function if no valid data

    #  STEP 2: Calculate aggregate statistics 
    combined_cop_data = np.vstack(all_cop_points)
    mean_trajectory = np.mean(resampled_trajectories, axis=0)

    # Calculation of the ellipse on total variability
    mean_cop_total = np.mean(combined_cop_data, axis=0)
    cov_matrix = np.cov(combined_cop_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues[eigenvalues < 0] = 0
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    vx, vy = eigenvectors[:, 0]
    angle = np.degrees(np.arctan2(vy, vx))
    chi2_val = chi2.ppf(0.95, 2)
    width = 2 * np.sqrt(chi2_val * eigenvalues[0])
    height = 2 * np.sqrt(chi2_val * eigenvalues[1])
    total_sway_area = np.pi * (width / 2) * (height / 2)


    #  STEP 3: Create the 2D Plot 
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(TRIANGLE_POINTS[:, 0], TRIANGLE_POINTS[:, 1], color='gray', linestyle='--', linewidth=1.5, label='Sensors Perimeter')
    ax.scatter(combined_cop_data[:, 0], combined_cop_data[:, 1], c='lightblue', s=5, alpha=0.4, label=f'Variability CoP (N={len(list_of_dfs)})')
    ax.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], color='darkblue', linewidth=3, label='Average Trajectory')
    ax.plot(mean_trajectory[0, 0], mean_trajectory[0, 1], 'o', markersize=12, markerfacecolor='lime', markeredgecolor='black', label='Average starting point')

  
    # STEP 4: Finalize and Save 
    ax.set_title(f'2D Task Analysis: {task_name.upper()} (Subject: {subject_id})', fontweight='bold')
    ax.set_xlabel('Medio-Lateral [cm]')
    ax.set_ylabel('Antero-Posterior [cm]')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), loc='best') 
    output_path = os.path.join(output_dir, f"{subject_id}_{task_name}_2d_summary_triangle.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary 2D plot saved to: {output_path}")

def main():
    """
    Main function that orchestrates plot creation.
    """
    if not os.path.exists(DATA_DIRECTORY):
        print(f"ERROR: Data directory '{DATA_DIRECTORY}' not found.")
        return
    if not os.path.exists(PLOTS_DIRECTORY):
        os.makedirs(PLOTS_DIRECTORY)

    organized_files = defaultdict(lambda: defaultdict(list))
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith(".csv"):
            parts = filename.replace(".csv", "").split('_')
            subject_id = parts[0]
            task_name = "_".join(parts[1:-1])
            filepath = os.path.join(DATA_DIRECTORY, filename)
            organized_files[subject_id][task_name].append(filepath)

    for subject_id, tasks in organized_files.items():
        for task_name, file_list in tasks.items():
            print(f"Analyzing 2D: {subject_id}, Task {task_name}...")
            try:
                list_of_dfs = [pd.read_csv(fp) for fp in file_list]
                plot_2d_summary(list_of_dfs, subject_id, task_name, PLOTS_DIRECTORY)
            except Exception as e:
                print(f"  !! Error: {e}")


if __name__ == '__main__':
    main()