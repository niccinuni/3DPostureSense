# 3DPostureSense

This repository contains the complete software ecosystem for a sensorized smart armrest project. The system is designed to acquire force data, analyze Center of Pressure (CoP) trajectories, and classify user movements (such as Sit-to-Stand, Leaning, and Sway) using Machine Learning.

## Repository Structure

The project is organized into the following modules:

### 1. `ESP32`
This directory contains the core data acquisition software:
*   **Firmware (C++/Arduino):** The code running on the ESP32 microcontroller. It handles:
    *   Signal acquisition from Hall effect sensors (or load cells).
    *   Real-time signal filtering (EMA - Exponential Moving Average).
    *   Two-stage calibration (Relative Sensitivity & Global Force Scale).
    *   Transmission of data via Serial (USB).
*   **Data Logger (Python):** A `PyQt5` graphical interface used to:
    *   Connect to the ESP32.
    *   Guide the subject through the experimental protocol (Sway, Lean Left/Right, STS, Tap).
    *   Visualize real-time data.
    *   Save trial data to CSV files.

### 2. `Armrest Model`
Contains mathematical modeling and calibration scripts:
*   **MATLAB:** Scripts used for the mechanical characterization of the sensors.
    *   Specifically, the **Voltage-to-Deformation** calibration model to translate sensor readings into physical displacement/force units based on robot-guided indentation tests.

### 3. `2D and 3D Analysis`
Python scripts for post-processing and visualizing the collected experimental data:
*   **2D Summary Plots:** Generates trajectory plots of the Center of Pressure (CoP) including 95% confidence ellipses and variability clouds.
*   **3D Surface Plots:** Visualizes the mean force distribution across the armrest surface as a smooth 3D mesh with gradient vectors.

### 4. `Classifier and validation`
Machine Learning pipeline for movement recognition:
*   **Feature Extraction:** Extracts temporal and spatial features from the raw CSV data (e.g., CoP displacement, Peak Force, RFD).
*   **Classification:** Implements a **Random Forest Classifier**.
*   **Validation:** Validates the model using a **Leave-One-Subject-Out (LOSO)** cross-validation strategy to ensure the system generalizes well to new users.


##  Getting Started

### Prerequisites
*   **Hardware:** ESP32 board, Hall effect sensors/Load cells, connection wires.
*   **Software:**
    *   Arduino IDE (for flashing the ESP32).
    *   Python 3.8+ (with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `PyQt5`, `pyqtgraph`).
    *   MATLAB (optional, for re-calibrating the model).

### Installation & Usage Workflow

1.  **Firmware Setup:**
    *   Navigate to the `ESP32` folder.
    *   Open the `.ino` file in Arduino IDE.
    *   Update the calibration constants (found in the MATLAB output) if necessary.
    *   Flash the code to your ESP32 board.

2.  **Data Acquisition:**
    *   Run the Python GUI script located in the `ESP32` folder.
    *   Enter the Subject ID and follow the on-screen protocol to record data.

3.  **Visualization:**
    *   Move the recorded CSV files to a data directory.
    *   Run the scripts in `2D and 3D Analysis` to generate visual reports of the trials.

4.  **Classification:**
    *   Run the script in `Classifier and validation` to train the model and evaluate accuracy using the collected dataset.

