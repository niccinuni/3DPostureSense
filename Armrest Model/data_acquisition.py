import serial
import sys
import time
import os
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np

SERIAL_PORT = "COM3"
BAUD_RATE = 115200
OUTPUT_DIRECTORY = "experimental_data_timed_Riccardo"


# Format: (task_name, number_of_repetitions, duration_in_seconds)
PROTOCOL = [
    ('sway',       1, 30.0),
    ('lean_right', 5, 7.0),  
    ('lean_left',  5, 7.0),
    ('sts',        5, 5.0),  
    ('tap',        5, 3.0)   
]

# INSTRUCTIONS 
INSTRUCTIONS = {
    'sway': "Task: RESTING STABILITY (SWAY)\n\n"
            "Instruction for subject: 'Sit still and relaxed, as motionless as possible.'\n\n"
            f"Recording will last {PROTOCOL[0][2]} seconds.",
    'lean_right': "Task: LEAN RIGHT\n\n"
                  "Instruction: 'When I press Start, lean SLOWLY to the RIGHT, hold, and return to center.'\n\n"
                  f"Recording will last {PROTOCOL[1][2]} seconds.",
    'lean_left': "Task: LEAN LEFT\n\n"
                 "Instruction: 'When I press Start, lean SLOWLY to the LEFT, hold, and return to center.'\n\n"
                 f"Recording will last {PROTOCOL[2][2]} seconds.",
    'sts': "Task: SIT-TO-STAND (STS)\n\n"
           "Instruction: 'When I press Start, stand up naturally.'\n\n"
           f"Recording will last {PROTOCOL[3][2]} seconds.",
    'tap': "Task: IMPULSE (TAP)\n\n"
           "Instruction: 'Rest your arm and, after I press Start, give a sharp tap on the armrest.'\n\n"
           f"Recording will last {PROTOCOL[4][2]} seconds."
}

class SerialReader(QtCore.QThread):
    newData = QtCore.pyqtSignal(list)
    def __init__(self):
        super().__init__(); self.ser = None; self.running = True
    def run(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            print(f"Connected to {SERIAL_PORT}..."); self.ser.flushInput()
            for _ in range(5): self.ser.readline()
        except serial.SerialException as e:
            print(f"Critical Serial Error: {e}"); self.running = False; return
        while self.running:
            if self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    if len(line.split(',')) == 9: self.newData.emit(line.split(','))
                except (UnicodeDecodeError, ValueError): pass
            time.sleep(0.001)
        if self.ser and self.ser.is_open: self.ser.close(); print("Serial connection closed.")
    def stop(self): self.running = False


class ExperimentApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.subject_id = ""; self.protocol_step = -1; self.repetition_count = 0
        self.is_recording = False; self.output_file = None
        # F_sx -> F_left, F_dx -> F_right
        self.csv_header = "Timestamp,F_left,F_right,F_vtc,F_tot,is_rested,copStateChanged,CoP_X,CoP_Y\n"
        self.recording_timer = QtCore.QTimer(self); self.recording_timer.setSingleShot(True)
        self.recording_timer.timeout.connect(self.stop_recording)
        
        self.setWindowTitle('Experimental Protocol Assistant (Timed)'); self.setGeometry(100, 100, 800, 600)
        main_layout = QtWidgets.QVBoxLayout(self)
        self.instruction_label = QtWidgets.QLabel("Welcome. Please enter a Subject ID."); main_layout.addWidget(self.instruction_label)
        self.status_label = QtWidgets.QLabel("Status: Waiting"); main_layout.addWidget(self.status_label)
        self.plot_widget = pg.PlotWidget(); self.plot_item = self.plot_widget.plot(pen=None, symbol='o', symbolSize=10, symbolBrush='r'); main_layout.addWidget(self.plot_widget)
        self.control_button = QtWidgets.QPushButton("Start Experiment"); self.control_button.clicked.connect(self.advance_protocol); main_layout.addWidget(self.control_button)
        
    
        self.instruction_label.setFont(QtGui.QFont('Arial', 12, QtGui.QFont.Bold)); self.instruction_label.setStyleSheet("padding: 10px; border: 1px solid #ccc; background-color: #f0f0f0;"); self.instruction_label.setWordWrap(True)
        self.status_label.setFont(QtGui.QFont('Arial', 10)); self.control_button.setFont(QtGui.QFont('Arial', 14, QtGui.QFont.Bold)); self.control_button.setMinimumHeight(50)
        
        self.serial_reader = SerialReader(); self.serial_reader.newData.connect(self.handle_new_data); self.serial_reader.start()

    def advance_protocol(self):
        if self.is_recording: return 
        
        if self.protocol_step == -1:
            text, ok = QtWidgets.QInputDialog.getText(self, 'Subject ID', 'Enter Subject ID:')
            if ok and text.strip():
                self.subject_id = text.strip(); os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
                self.protocol_step = 0; self.repetition_count = 1
                self.update_ui_for_current_task()
            return

        _, total_reps, _ = PROTOCOL[self.protocol_step]
        if self.repetition_count < total_reps: self.repetition_count += 1
        else: self.protocol_step += 1; self.repetition_count = 1
        
        if self.protocol_step >= len(PROTOCOL): self.show_completion_message()
        else: self.update_ui_for_current_task()

    def update_ui_for_current_task(self):
        task_name, total_reps, _ = PROTOCOL[self.protocol_step]
        self.instruction_label.setText(INSTRUCTIONS[task_name])
        self.status_label.setText(f"Ready for: {task_name.upper()} | Repetition: {self.repetition_count} of {total_reps}")
        self.control_button.setText("▶ Start Recording")
        self.control_button.setStyleSheet("background-color: #A8E6A8;"); self.control_button.setEnabled(True)
        try: self.control_button.clicked.disconnect() 
        except TypeError: pass
        self.control_button.clicked.connect(self.start_recording)

    def start_recording(self):
        if self.is_recording: return

        task_name, _, duration = PROTOCOL[self.protocol_step]
        filename = os.path.join(OUTPUT_DIRECTORY, f"{self.subject_id}_{task_name}_rep{self.repetition_count}.csv")
        
        try:
            self.output_file = open(filename, 'w', newline='')
            self.output_file.write(self.csv_header); self.is_recording = True
            
            self.status_label.setText(f"REC ● | Recording for {duration}s... -> {os.path.basename(filename)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            
            
            self.control_button.setText("Recording in Progress...")
            self.control_button.setEnabled(False)
            self.recording_timer.start(int(duration * 1000))

        except IOError as e: self.status_label.setText(f"File Error: {e}")

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False
        if self.output_file: self.output_file.close(); self.output_file = None
        
        self.status_label.setStyleSheet("color: #006400; font-weight: bold;")
        self.status_label.setText("Recording completed and saved.")
        
        self.control_button.setText("Next Task →")
        self.control_button.setStyleSheet(""); self.control_button.setEnabled(True)
        try: self.control_button.clicked.disconnect()
        except TypeError: pass
        self.control_button.clicked.connect(self.advance_protocol)

    def handle_new_data(self, data_parts: list):
        if self.is_recording and self.output_file:
            self.output_file.write(",".join(data_parts) + "\n")
        try:
            cop_x_str, cop_y_str = data_parts[-2], data_parts[-1]
            cop_x = float(cop_x_str) if cop_x_str != 'nan' else np.nan
            cop_y = float(cop_y_str) if cop_y_str != 'nan' else np.nan
            if not np.isnan(cop_x) and not np.isnan(cop_y): self.plot_item.setData([cop_x], [cop_y])
        except (ValueError, IndexError): pass

    def show_completion_message(self):
        self.instruction_label.setText("Protocol completed for subject " + self.subject_id)
        self.status_label.setText("All data has been saved. Thank you!")
        self.control_button.setText("Experiment Finished"); self.control_button.setEnabled(False)
        
    def closeEvent(self, event: QtGui.QCloseEvent):
        if self.is_recording: self.stop_recording()
        self.serial_reader.stop(); self.serial_reader.wait(); event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ExperimentApp()
    window.show()
    sys.exit(app.exec_())