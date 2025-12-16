
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from collections import defaultdict
from pathlib import Path

plt.rcParams.update({
    "font.size": 20,        
    "axes.titlesize": 17,   
    "axes.labelsize": 17,   
    "xtick.labelsize": 17,  
    "ytick.labelsize": 17,  
    "legend.fontsize": 15   
    })

DATA_DIRECTORY = Path("experimental_data_timed_Niconew") 
PLOTS_DIRECTORY = Path("surface_plots")
INTERPOLATION_POINTS = 101
GRID_RESOLUTION = 50

#  SYSTEM GEOMETRY 
P_SX = np.array([0.0, 0.0])   
P_DX = np.array([7.00, 0.0])  
P_VTC = np.array([3.5, 24.5]) 
TRIANGLE_POINTS = np.array([P_SX, P_DX, P_VTC, P_SX])

def plot_mean_force_surface(list_of_dfs: list, subject_id: str, task_name: str, output_dir: Path):
    """
    Analyzes mean task data and generates a 3D force surface plot.
    """
    resampled_data, all_cop_points = [], []

    #Calculate mean signals 
    for df in list_of_dfs:
        df.replace('nan', np.nan, inplace=True); df.dropna(subset=['CoP_X', 'CoP_Y', 'F_sx', 'F_dx', 'F_vtc'], inplace=True)
        df = df.astype({c: float for c in df.columns if c not in ['Timestamp', 'is_rested', 'copStateChanged']})
        df = df[df['F_tot'] > 1.0].copy(); 
        if len(df) < 20: continue
        all_cop_points.append(df[['CoP_X', 'CoP_Y']].values)
        df['Time_s'] = (df['Timestamp'] - df['Timestamp'].iloc[0]) / 1000.0
        t_new = np.linspace(df['Time_s'].iloc[0], df['Time_s'].iloc[-1], INTERPOLATION_POINTS)
        resampled_data.append(np.vstack((
            np.interp(t_new, df['Time_s'], df['F_sx']), np.interp(t_new, df['Time_s'], df['F_dx']),
            np.interp(t_new, df['Time_s'], df['F_vtc']), np.interp(t_new, df['Time_s'], df['CoP_X']),
            np.interp(t_new, df['Time_s'], df['CoP_Y']),
        )).T)
    if not resampled_data: print(f"Skipping {subject_id}_{task_name}: insufficient data."); return

    mean_signals = np.mean(resampled_data, axis=0)
    mean_f_sx, mean_f_dx, mean_f_vtc, mean_cop_x, mean_cop_y = mean_signals.T
    mean_f_tot = mean_f_sx + mean_f_dx + mean_f_vtc
    peak_idx = np.argmax(mean_f_tot)
    peak_forces = {'F_sx': mean_f_sx[peak_idx], 'F_dx': mean_f_dx[peak_idx], 'F_vtc': mean_f_vtc[peak_idx]}
    peak_cop = {'x': mean_cop_x[peak_idx], 'y': mean_cop_y[peak_idx]}

    #  Reconstruct force surface and gradient 
    known_points = np.array([P_SX, P_DX, P_VTC])
    known_values = np.array([peak_forces['F_sx'], peak_forces['F_dx'], peak_forces['F_vtc']])
    
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 8, GRID_RESOLUTION), np.linspace(-1, 26, GRID_RESOLUTION))
    grid_z = griddata(known_points, known_values, (grid_x, grid_y), method='cubic', fill_value=0)
    grid_z[grid_z < 0] = 0
    gy, gx = np.gradient(grid_z)

    # Creation of Improved 3D Plot 
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(12, 12)); ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Draw the 3D SMOOTH SURFACE
    cmap = 'viridis'
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cmap, alpha=0.9, rstride=1, cstride=1, 
                           edgecolor='none', antialiased=True, zorder=2)
    
    # Add a colorbar for force
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1, label='Interpolated Force [N]')
    
    # Draw GRADIENT VECTORS on the floor
    skip = 5
    ax.quiver(grid_x[::skip, ::skip], grid_y[::skip, ::skip], 0, 
              gx[::skip, ::skip], gy[::skip, ::skip], 0,
              color='black', length=0.5, normalize=True, zorder=3, label='Force Gradient')

    # Draw REFERENCES on the floor
    ax.plot(TRIANGLE_POINTS[:, 0], TRIANGLE_POINTS[:, 1], 0, '--', color='red', linewidth=2, zorder=4, label='Sensor Perimeter')
    ax.scatter(known_points[:, 0], known_points[:, 1], 0, c='red', s=100, zorder=5, label='Sensor position')
    ax.scatter(peak_cop['x'], peak_cop['y'], 0, c='red', s=200, marker='X', linewidth=3, zorder=5, label='Average CoP at Peak')
    
    
    #ax.set_title(f'Mean Task Analysis: {task_name.upper()}\n(Subject: {subject_id} | N={len(list_of_dfs)})', fontweight='bold', fontsize=16)
    ax.set_xlabel('\n ML Axis[cm]'); ax.set_ylabel('\n AP Axis [cm]'); ax.set_zlabel('\nForce [N]')
    ax.view_init(elev=40, azim=-70)
    
    
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    
    ax.legend(loc='upper left')

    output_path = output_dir / f"{subject_id}_{task_name}_surface.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"  Surface plot saved to: {output_path}")


def main():
    print(" Starting Force Surface Plot Generation ")
    PLOTS_DIRECTORY.mkdir(exist_ok=True)

    organized_files = defaultdict(lambda: defaultdict(list))
    if not DATA_DIRECTORY.exists(): print(f"ERROR: Data directory '{DATA_DIRECTORY}' not found."); return
        
    for filepath in DATA_DIRECTORY.glob("*.csv"):
        try:
            parts = filepath.stem.split('_'); subject_id, task_name = parts[0], "_".join(parts[1:-1])
            organized_files[subject_id][task_name].append(filepath)
        except IndexError: print(f"Warning: file ignored: {filepath.name}")

    if not organized_files: print("No CSV files found."); return

    for subject_id, tasks in organized_files.items():
        for task_name, file_list in tasks.items():
            print(f"Analyzing: {subject_id}, Task {task_name}...")
            try:
                list_of_dfs = [pd.read_csv(fp) for fp in file_list]
                plot_mean_force_surface(list_of_dfs, subject_id, task_name, PLOTS_DIRECTORY)
            except Exception as e:
                print(f"  !! Error: {e}"); import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()