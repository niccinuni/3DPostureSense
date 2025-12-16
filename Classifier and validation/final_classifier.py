import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List
from pathlib import Path


DATA_DIRECTORY = Path("data")
OUTPUT_DIRECTORY = Path("classification_results")

def extract_normalized_features(df: pd.DataFrame, task_name: str) -> Dict[str, float]:
    """
    Extracts a set of "smart" and normalized features relative to the
    starting position to be invariant to the initial offset.
    """
    features = {}
    df.replace('nan', np.nan, inplace=True); df.dropna(subset=['CoP_X', 'CoP_Y'], inplace=True)
    df = df.astype({c: float for c in df.columns if c not in ['Timestamp', 'is_rested', 'copStateChanged']})
    df = df[df['F_tot'] > 1.0].copy(); 
    if len(df) < 20: return None
    df['Time_s'] = (df['Timestamp'] - df['Timestamp'].iloc[0]) / 1000.0
    
    start_point_x, start_point_y = df['CoP_X'].iloc[0], df['CoP_Y'].iloc[0]
    df['CoP_X_relative'] = df['CoP_X'] - start_point_x
    df['CoP_Y_relative'] = df['CoP_Y'] - start_point_y
    
    features['cop_mean_x_rel'] = df['CoP_X_relative'].mean()
    features['cop_max_x_rel'] = df['CoP_X_relative'].max()
    features['cop_min_x_rel'] = df['CoP_X_relative'].min()
    features['cop_mean_y_rel'] = df['CoP_Y_relative'].mean()
    features['cop_displacement_x'] = df['CoP_X_relative'].max() - df['CoP_X_relative'].min()
    features['cop_displacement_y'] = df['CoP_Y_relative'].max() - df['CoP_Y_relative'].min()
    features['peak_force'] = df['F_tot'].max()
    features['duration_s'] = df['Time_s'].iloc[-1]
    features['cop_path_length'] = np.sum(np.sqrt(np.sum(np.diff(df[['CoP_X_relative', 'CoP_Y_relative']].values, axis=0)**2, axis=1)))
    
    if 'sts' in task_name:
        peak_force_idx = df['F_tot'].idxmax()
        time_to_peak = df.loc[peak_force_idx, 'Time_s'] if df.loc[peak_force_idx, 'Time_s'] > 0 else 1e-6
        features['rfd'] = features['peak_force'] / time_to_peak
    else: features['rfd'] = 0.0
    if 'tap' in task_name:
        time_diff = np.diff(df['Time_s']); force_diff = np.diff(df['F_tot'])
        force_derivative = force_diff[time_diff > 0] / time_diff[time_diff > 0]
        features['peak_force_derivative'] = np.max(np.abs(force_derivative)) if len(force_derivative) > 0 else 0.0
    else: features['peak_force_derivative'] = 0.0
    return features

def main():
    """
    Main function orchestrating the classification test.
    """
    print(" Starting Final Classifier (Mean-Level LOSO CV) ")
    
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    
    #  1. Loading and feature extraction 
    all_features = []
    if not DATA_DIRECTORY.exists(): print(f"ERROR: Directory '{DATA_DIRECTORY}' not found."); return
        
    for filepath in DATA_DIRECTORY.glob("*.csv"):
        try:
            parts = filepath.stem.split('_'); subject, task, rep = parts[0], "_".join(parts[1:-1]), parts[-1]
            df = pd.read_csv(filepath)
            features = extract_normalized_features(df, task)
            if features:
                features['subject'] = subject; features['task'] = task
                all_features.append(features)
        except (ValueError, IndexError): print(f"Warning: file ignored: {filepath.name}")

    if not all_features: print("No valid data found."); return
    
    trial_level_df = pd.DataFrame(all_features)

    # 2. Mean-Level Aggregation 
    mean_level_df = trial_level_df.groupby(['subject', 'task']).mean().reset_index()
    
    num_subjects = mean_level_df['subject'].nunique()
    if num_subjects < 2:
        print(f"\nERROR: Only {num_subjects} subject found. Leave-One-Subject-Out validation requires at least 2 subjects.")
        return

    print(f"\nCreated {len(mean_level_df)} mean motor profiles for {num_subjects} subjects.")
    
    #  3. Data preparation for scikit-learn 
    feature_columns = [col for col in mean_level_df.columns if col not in ['subject', 'task']]
    X = mean_level_df[feature_columns]
    y = mean_level_df['task']
    groups = mean_level_df['subject']
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    logo = LeaveOneGroupOut()
    
    print(f"\nRunning Leave-One-Subject-Out Cross-Validation...")
    
    #  4. Execution and Results Calculation 
    y_pred = cross_val_predict(pipeline, X, y, cv=logo, groups=groups)
    accuracy = accuracy_score(y, y_pred)
    
    print("\n-------------------------------------------------------------")
    print(f"Average Generalization Accuracy (LOSO CV): {accuracy * 100:.2f}%")
    print("-------------------------------------------------------------\n")

    #  5. Confusion Matrix 
    labels = sorted(mean_level_df['task'].unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    
    cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Predicted: {l}" for l in labels])
    report_path = OUTPUT_DIRECTORY / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Average Generalization Accuracy (LOSO CV): {accuracy * 100:.2f}%\n\n")
        f.write("Confusion Matrix (aggregated on each fold):\n")
        f.write(cm_df.to_string())
    print(f"Classification report saved to: {report_path}")

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 10)) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    

    ax.set_title('Confusion Matrix (Leave-One-Subject-Out CV)', fontweight='bold', fontsize=20)
    

    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)

    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)

    for text_array in disp.text_:
        for text in text_array:
            text.set_fontsize(14)


    plt.tight_layout() 
    cm_path = OUTPUT_DIRECTORY / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight') 
    plt.close(fig)
    print(f"Visual confusion matrix saved to: {cm_path}")
    
    print("\n Analysis Completed ")

if __name__ == '__main__':
    main()