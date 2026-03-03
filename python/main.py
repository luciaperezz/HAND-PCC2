# main.py
# This file is the entry point for Phase 1
# It calls all functions: reading, processing, extraction, rules, and visualization

import asyncio
from collections import deque
from enum import Enum

#---------------------------------------------------
#           IMPORT MODULES
#---------------------------------------------------

import signal_processing
import feature_extraction
import parser
import serial_reader
import rules_feedback

# Buffers
buffer_pressure = deque(maxlen=200)
buffer_imu_acc = deque(maxlen=200)
buffer_imu_gyro = deque(maxlen=200)


#---------------------------------------------------
#           DEFINE ENUMS
#---------------------------------------------------
class State(Enum):
    IDLE = 0
    CALIBRATION = 1
    EXERCISE_1 = 2
    EXERCISE_2 = 3
    REST = 4
    END = 5

class Disease(Enum):
    TYPE_STROKE = 0
    TYPE_TREMOR = 1

#---------------------------------------------------
#           DEFINE HELPER FUNCTIONS
#---------------------------------------------------
def preprocess_data(raw_data, current_disease):
    """
    raw_data: dict with keys like 'pressure', 'emg', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'
    returns: dict with processed signals or features ready for extraction
    """
    processed = {}

    #STROKE- EXERCISE 1: Pressue + EMG 
    #STROKE- EXERCISE 2: Pressue + EMG

    #TREMOR- EXERCISE 1: Pressue + IMU
    #TREMOR- EXERCISE 2: Pressue + IMU

    # Pressure always processed
    processed['pressure'] = signal_processing.process_pressure(raw_data['pressure'])

    # EMG only for stroke patients 
    if current_disease == Disease.TYPE_STROKE and 'emg' in raw_data:
        processed['emg'] = signal_processing.process_emg(raw_data['emg'])

    # IMU only for tremor patients
    if current_disease == Disease.TYPE_TREMOR and all(k in raw_data for k in ['ax','ay','az','gx','gy','gz']):
        processed['imu'] = signal_processing.process_imu(
            raw_data['ax'], raw_data['ay'], raw_data['az'],
            raw_data['gx'], raw_data['gy'], raw_data['gz']
        )

    return processed

#------------------ CONTROL FUNCITONS-------------------------------
def get_user_choice():
    choice = input("Choose exercise (1 or 2): ")
    if choice == "1":
        return State.EXERCISE_1
    else:
        return State.EXERCISE_2
    
def need_recalibration():
    return False  # simple

#------------------ EXERCISES LOGIC -------------------------------
def calibrate():
    print("Calibrating...")
    # calibrating LOGIC
    return 1234  # MVC fake de momento

def do_exercise_1_stroke(data, mvc):
    features = feature_extraction.extract_features_ex1_stroke(data, mvc)
    feedback = rules_feedback.evaluate(features)
    print("EX1 Stroke feedback:", feedback)

def do_exercise_2_stroke(data, mvc):
    features = feature_extraction.extract_features_ex2_stroke(data, mvc)
    feedback = rules_feedback.evaluate(features)
    print("EX2 Stroke feedback:", feedback)

def do_exercise_1_tremor(data, mvc):
    features = feature_extraction.extract_features_ex1_tremor(data, mvc)
    feedback = rules_feedback.evaluate(features)
    print("EX1 Tremor feedback:", feedback)

def do_exercise_2_tremor(data, mvc):
    features = feature_extraction.extract_features_ex2_tremor(data, mvc)
    feedback = rules_feedback.evaluate(features)
    print("EX2 Tremor feedback:", feedback)

#------------------ CONDITIONS FUNCITONS-------------------------------

def exercise_1_finished():
    return False

def exercise_2_finished():
    return False 

def rest_finished():
    return True

#---------------------------------------------------
#           MAIN LOOP: READ - PROCESS - FSM
#---------------------------------------------------
async def main():

    reader = serial_reader.BLEReader()
    await reader.connect()
    
    state = State.IDLE
    current_disease = Disease.TYPE_STROKE
    mvc = None
    selected_exercise = None

    while True:

        #Read raw sensor data
        raw_line = await serial_reader.read_line()

        #Parse to structured format
        raw_data = parser.parse_line(raw_line)
        # data = {'pressure': 1875, 'gsr': 2020, 'moisture': 1550}
        
        if raw_data is None:
            continue

        #Signal processing
        data = preprocess_data(raw_data, current_disease)  
        # e.g., data['pressure'] = filtered value, data['gsr'] = normalized value

        #---------------FSM--------------

        #------------------------------------------------------------------------------------------------
        #                           IDLE
        #-------------------------------------------------------------------------------------------------
        if state == State.IDLE:
            selected_exercise = get_user_choice()
            
            if mvc is None:
                print("Waiting to start...")
                state = State.CALIBRATION
            else:
                state = selected_exercise

        #-------------------------------------------------------------------------------------------------------
        #                           CALIBRATION
        #-------------------------------------------------------------------------------------------------------
        elif state == State.CALIBRATION:
            mvc = calibrate(buffer_pressure)
            state = selected_exercise
        
        #-------------------------------------------------------------------------------------------------------
        #                           EXERCISE 1
        #-------------------------------------------------------------------------------------------------------
        elif state == State.EXERCISE_1:
            if current_disease == Disease.TYPE_STROKE:
                do_exercise_1_stroke(data, mvc)
            else:
                do_exercise_1_tremor(data, mvc)

            if exercise_1_finished():
                state = State.REST


        #-------------------------------------------------------------------------------------------------------
        #                           EXERCISE 2
        #-------------------------------------------------------------------------------------------------------
        elif state == State.EXERCISE_2:
            if current_disease == Disease.TYPE_STROKE:
                do_exercise_2_stroke(data, mvc)
            elif current_disease == Disease.TREMOR:
                do_exercise_2_tremor(data, mvc)
            if exercise_2_finished():
                state = State.REST


        #-------------------------------------------------------------------------------------------------------
        #                           REST
        #-------------------------------------------------------------------------------------------------------
        elif state == State.REST:
            if rest_finished():
                selected_exercise = get_user_choice()
                if need_recalibration():
                    mvc = None
                    state = State.CALIBRATION
                else:
                    state = selected_exercise




if __name__ == "__main__":
    asyncio.run(run_ble(on_data))