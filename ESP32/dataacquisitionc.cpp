
//  LIBRARIES 
#include <AS5600.h>
#include <Wire.h>

//  LOGGING PARAMETERS 
#define LOG_INTERVAL_MS 50
unsigned long lastLogTime = 0;


//  HARDWARE CONFIGURATION

const int hallPin_VTC   = 34; // Vertex/Top Sensor
const int hallPin_Left  = 32; // Left Sensor (SX)
const int hallPin_Right = 35; // Right Sensor (DX)


//  MODELS, GEOMETRY, AND CALIBRATION PARAMETERS


//  MODEL 1: VOLTAGE -> "RAW UNITS" (Transfer Function) 
// This polynomial converts voltage into an intermediate unit proportional to force.
const float P1 = -2.13971683f;
const float P2 = 18.96856857f;
const float P3 = -25.61022289f;

//  STAGE 1: RELATIVE SENSITIVITY COEFFICIENTS 
// Obtained by placing a known load on each sensor and normalizing
// the responses relative to a reference sensor.
const float C_REL_LEFT  = 1.0f;          // Reference sensor 
const float C_REL_RIGHT = 23.0/26.0f;    /
const float C_REL_VTC   = 23.0/32.4f;    

//  STAGE 2: GLOBAL SYSTEM SCALE FACTOR 
// Obtained by placing a known load (3 kg) at the center of the system,
// (3.0 * 9.81) / F_tot_raw_corrected
const float SYSTEM_FORCE_SCALE = 5.91f;

//  FILTERING PARAMETERS (EMA - Exponential Moving Average) 
const float EMA_ALPHA = 0.15f; // Smoothing factor (0 < alpha < 1). Lower = smoother.
float filtered_v_left, filtered_v_right, filtered_v_vtc;

//  VARIABLES FOR AUTO-ZEROING 
float v_rest_left = 0.0, v_rest_right = 0.0, v_rest_vtc = 0.0;
const int ZEROING_SAMPLES = 200;

//  SENSOR GEOMETRY (cm) 
const float P_LEFT_XY[] = {0.0, 0.0};
const float P_RIGHT_XY[] = {7.00, 0.0};
const float P_VTC_XY[] = {3.5, 22.0};
const float GEOMETRIC_CENTER_XY[] = {
    (P_LEFT_XY[0] + P_RIGHT_XY[0] + P_VTC_XY[0]) / 3.0, 
    (P_LEFT_XY[1] + P_RIGHT_XY[1] + P_VTC_XY[1]) / 3.0
};

//  CONTROL PARAMETERS (to be justified experimentally) 
// Threshold determined as N times the standard deviation of noise at rest.
const float MIN_REST_THRESHOLD_N = 3.5; 
// Radius based on maximum CoP fluctuation under stable load.
const float DEAD_ZONE_RADIUS_CM = 2.0;

//  State Variables 
bool was_in_deadzone = false;


//ùSETUP

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\nStarting standardized sketch (v8.1)...");
    
    Wire.begin(21, 22);

    //  AUTO-ZEROING PROCEDURE ON VOLTAGE 
    Serial.println("Starting auto-zeroing. Do not touch the system...");
    float sum_v_left = 0, sum_v_right = 0, sum_v_vtc = 0;
    for (int i = 0; i < ZEROING_SAMPLES; i++) {
        sum_v_left  += analogRead(hallPin_Left)  * (3.3 / 4095.0);
        sum_v_right += analogRead(hallPin_Right) * (3.3 / 4095.0);
        sum_v_vtc   += analogRead(hallPin_VTC)   * (3.3 / 4095.0);
        delay(10);
    }
    v_rest_left  = sum_v_left  / ZEROING_SAMPLES;
    v_rest_right = sum_v_right / ZEROING_SAMPLES;
    v_rest_vtc   = sum_v_vtc   / ZEROING_SAMPLES;
    
    // Initialize filters with rest values to avoid initial transient
    filtered_v_left  = v_rest_left;
    filtered_v_right = v_rest_right;
    filtered_v_vtc   = v_rest_vtc;

    Serial.println("Auto-zeroing completed.");
    Serial.print("Rest Voltages [V]: Left="); Serial.print(v_rest_left, 4);
    Serial.print(", Right="); Serial.print(v_rest_right, 4);
    Serial.print(", VTC="); Serial.println(v_rest_vtc, 4);
    
    Serial.println("System ready.");
    // Header updated to English to match Python scripts (F_sx -> F_left, F_dx -> F_right)
    Serial.println("Timestamp,F_left,F_right,F_vtc,F_tot,is_rested,copStateChanged,CoP_X,CoP_Y");
}


//  HELPER FUNCTIONS

float voltageToRawUnit(float voltage, float v_rest) {
    // Calculate voltage variation, ignore negative values
    if (voltage <= v_rest) {
      return 0.0f;
    }
    //  model to ABSOLUTE voltage
    float raw_estimate = (P1 * voltage * voltage) + (P2 * voltage) + P3;
    return max(0.0f, raw_estimate);
}



// MAIN LOOP

void loop() {
    // 1. SIGNAL ACQUISITION AND FILTERING
    float v_left_raw  = analogRead(hallPin_Left)  * (3.3 / 4095.0);
    float v_right_raw = analogRead(hallPin_Right) * (3.3 / 4095.0);
    float v_vtc_raw   = analogRead(hallPin_VTC)   * (3.3 / 4095.0);

    //EMA filter to each voltage reading
    filtered_v_left  = (EMA_ALPHA * v_left_raw)  + (1.0 - EMA_ALPHA) * filtered_v_left;
    filtered_v_right = (EMA_ALPHA * v_right_raw) + (1.0 - EMA_ALPHA) * filtered_v_right;
    filtered_v_vtc   = (EMA_ALPHA * v_vtc_raw)   + (1.0 - EMA_ALPHA) * filtered_v_vtc;

    // 2. CONVERSION TO RAW UNITS
    float raw_left  = voltageToRawUnit(filtered_v_left,  v_rest_left);
    float raw_right = voltageToRawUnit(filtered_v_right, v_rest_right);
    float raw_vtc   = voltageToRawUnit(filtered_v_vtc,   v_rest_vtc);
    
    // 3.  TWO-STAGE CALIBRATION MODEL
    // Relative sensitivity correction
    float F_left_corr  = raw_left  * C_REL_LEFT;
    float F_right_corr = raw_right * C_REL_RIGHT;
    float F_vtc_corr   = raw_vtc   * C_REL_VTC;

    //  global scale factor to obtain Newtons
    float F_left  = F_left_corr  * SYSTEM_FORCE_SCALE;
    float F_right = F_right_corr * SYSTEM_FORCE_SCALE;
    float F_vtc   = F_vtc_corr   * SYSTEM_FORCE_SCALE;
    
    float F_tot = F_left + F_right + F_vtc;
    
    // 4. STATE CALCULATION, COP, AND DEADZONE LOGIC
    bool isArmRested = (F_tot > MIN_REST_THRESHOLD_N);
    
    float cop_x = NAN;
    float cop_y = NAN;
    bool copStateChanged = false;
    bool now_in_deadzone = false;

    if (isArmRested) {
        // Calculate CoP using final, corrected, and scaled forces
        cop_x = (P_LEFT_XY[0] * F_left + P_RIGHT_XY[0] * F_right + P_VTC_XY[0] * F_vtc) / F_tot;
        cop_y = (P_LEFT_XY[1] * F_left + P_RIGHT_XY[1] * F_right + P_VTC_XY[1] * F_vtc) / F_tot;
        
        float dx = cop_x - GEOMETRIC_CENTER_XY[0];
        float dy = cop_y - GEOMETRIC_CENTER_XY[1];
        float dist_from_center = sqrt(dx * dx + dy * dy);
        
        now_in_deadzone = (dist_from_center <= DEAD_ZONE_RADIUS_CM);
        
        // State change is detected only if the previous state was different
        if (now_in_deadzone != was_in_deadzone) {
            copStateChanged = true;
        }
        was_in_deadzone = now_in_deadzone;
    } else {
        // If the arm is not rested, CoP is undefined and not in any zone
        was_in_deadzone = false; 
    }

    // 5. DATA LOGGING
    if (millis() - lastLogTime >= LOG_INTERVAL_MS) {
        lastLogTime = millis();
        Serial.print(millis());         Serial.print(",");
        Serial.print(F_left, 3);        Serial.print(",");
        Serial.print(F_right, 3);       Serial.print(",");
        Serial.print(F_vtc, 3);         Serial.print(",");
        Serial.print(F_tot, 3);         Serial.print(",");
        Serial.print(isArmRested);      Serial.print(",");
        Serial.print(copStateChanged);  Serial.print(",");
        Serial.print(cop_x, 2);         Serial.print(",");
        Serial.println(cop_y, 2);
    }
}