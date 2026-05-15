# main.py - CORRECTED VERSION WITH FSM
# This file orchestrates the entire system

import asyncio
import threading
import sys
from collections import deque
from enum import Enum
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# IMPORTS OF OWN MODULES
# ═══════════════════════════════════════════════════════════════

import signal_processing
import feature_extraction
import parser
# import serial_reader  # Not used in Option A
# import rules_feedback  # TODO: implement

# ═══════════════════════════════════════════════════════════════
# START FASTAPI IN BACKGROUND
# ═══════════════════════════════════════════════════════════════

def start_api_server():
    """Starts FastAPI (api.py) in separate thread"""
    import uvicorn
    import os
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    uvicorn.run("web.api:app", host="0.0.0.0", port=8000, log_level="info")

def init_system():
    """Initializes the complete system"""
    print("🚀 Starting Smart Hand Rehabilitation System...")
    
    # Start FastAPI in background
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    print("✅ FastAPI started at: http://localhost:8000")
    print("🌐 WebSocket available at: ws://localhost:8000/ws")
    print("📡 Waiting for browser connection...\n")
    
    # Wait for FastAPI to start
    import time
    time.sleep(2)

# ═══════════════════════════════════════════════════════════════
# FSM STATES
# ═══════════════════════════════════════════════════════════════

class State(Enum):
    IDLE = "IDLE"                          # Waiting for login
    SESSION_ACTIVE = "SESSION_ACTIVE"      # User logged in
    TRIAL_RECORDING = "TRIAL_RECORDING"    # Recording data
    TRIAL_PROCESSING = "TRIAL_PROCESSING"  # Extracting features
    TRIAL_REST = "TRIAL_REST"              # Rest between trials
    EXERCISE_COMPLETE = "EXERCISE_COMPLETE"  # Exercise finished
    SESSION_COMPLETE = "SESSION_COMPLETE"  # Session finished

class Disease(Enum):
    TYPE_STROKE = 1
    TYPE_TREMOR = 2

# ═══════════════════════════════════════════════════════════════
# VALID TRANSITIONS MATRIX
# ═══════════════════════════════════════════════════════════════

VALID_TRANSITIONS = {
    State.IDLE: {
        "login": State.SESSION_ACTIVE,
    },
    State.SESSION_ACTIVE: {
        "start_trial": State.TRIAL_RECORDING,
        "change_exercise": State.SESSION_ACTIVE,
        "logout": State.IDLE,
        "reset_exercise": State.SESSION_ACTIVE
    },
    State.TRIAL_RECORDING: {
        "stop_trial": State.SESSION_ACTIVE,
        "timeout": State.TRIAL_PROCESSING,
        "reset_exercise": State.SESSION_ACTIVE
    },
    State.TRIAL_PROCESSING: {
        "features_ready": State.TRIAL_REST,
    },
    State.TRIAL_REST: {
        "rest_complete": State.SESSION_ACTIVE, 
        "exercise_complete": State.EXERCISE_COMPLETE,
        "reset_exercise": State.SESSION_ACTIVE,
        "stop_trial": State.SESSION_ACTIVE
    },
    State.EXERCISE_COMPLETE: {
        "next_exercise": State.SESSION_ACTIVE,
        "change_exercise": State.SESSION_ACTIVE,
        "logout": State.SESSION_COMPLETE,
        "login": State.SESSION_ACTIVE,
    },
    State.SESSION_COMPLETE: {
        "login": State.SESSION_ACTIVE,
    },
}

# ═══════════════════════════════════════════════════════════════
# FSM CLASS
# ═══════════════════════════════════════════════════════════════

class SessionFSM:
    def __init__(self):
        self.state = State.IDLE
        self.context = {
            "patient_id": None,
            "patient_group": None,
            "exercise": None,
            "trial_num": 0,
            "total_trials": 4,
            "trial_data": [],
            "trial_start_time": None,
            "rest_start_time": None,
            "mvc": None,  # Maximum from calibration
        }
    
    def transition(self, event, payload=None):
        """
        Attempt to transition to new state
        
        Args:
            event: event name (e.g., "start_trial")
            payload: additional event data
        
        Returns:
            True if valid transition, False if rejected
        """
        allowed = VALID_TRANSITIONS.get(self.state, {})
        
        if event not in allowed:
            print(f"❌ INVALID TRANSITION: {self.state.value} -[{event}]→ (not allowed)")
            return False
        
        next_state = allowed[event]
        prev_state = self.state
        
        print(f"✅ FSM: {prev_state.value} -[{event}]→ {next_state.value}")
        
        # Change state
        self.state = next_state
        
        # Execute actions when entering new state
        self._on_enter_state(next_state, payload or {})
        
        return True
    
    def _on_enter_state(self, state, payload):
        """Hook executed when entering a state"""
        if state == State.SESSION_ACTIVE:
            print(f"📋 Active session: {self.context['patient_id']}")
        
        elif state == State.TRIAL_RECORDING:
            self.context["trial_data"] = []
            self.context["trial_start_time"] = datetime.now()
            print(f"🎬 Trial {self.context['trial_num'] + 1} recording...")
        
        elif state == State.TRIAL_PROCESSING:
            print(f"⚙️  Processing trial features...")
        
        elif state == State.TRIAL_REST:
            self.context["rest_start_time"] = datetime.now()
            print(f"😴 Resting...")
        
        elif state == State.IDLE:
            # Reset context
            self.context = {
                "patient_id": None,
                "patient_group": None,
                "exercise": None,
                "trial_num": 0,
                "total_trials": 4,
                "trial_data": [],
                "mvc": None,
            }
            print("💤 System in IDLE")

# ═══════════════════════════════════════════════════════════════
# FEATURE PROCESSING
# ═══════════════════════════════════════════════════════════════

def extract_trial_features(trial_data, disease_type, mvc):
    """
    Extracts features from recorded trial
    
    Args:
        trial_data: list of frames {"pressure": ..., "emg": ..., "imu": {...}}
        disease_type: Disease.TYPE_STROKE or Disease.TYPE_TREMOR
        mvc: MVC calibration value
    
    Returns:
        dict with extracted features
    """
    if not trial_data:
        return {}
    
    # Extract arrays for each signal
    pressures = [f["pressure"] for f in trial_data if "pressure" in f]
    emg_raw = [f.get("emg_raw", 0) for f in trial_data]
    
    # Basic pressure features (always)
    features = {
        "pressure": {
            "mvc": max(pressures) if pressures else 0,
            "mean": sum(pressures) / len(pressures) if pressures else 0,
            "min": min(pressures) if pressures else 0,
        },
        "frames_captured": len(trial_data),
        "duration_secs": len(trial_data) / 25,  # Assuming 25 Hz
    }
    
    # Specific features by patient type
    if disease_type == Disease.TYPE_STROKE:
        # EMG features
        if emg_raw:
            features["emg"] = {
                "rms": (sum([e**2 for e in emg_raw]) / len(emg_raw))**0.5,
                "mean": sum(emg_raw) / len(emg_raw),
            }
        
        # RFD (Rate of Force Development)
        if len(pressures) > 1:
            features["pressure"]["rfd"] = (max(pressures) - pressures[0]) / len(pressures)
    
    elif disease_type == Disease.TYPE_TREMOR:
        # IMU features - TODO: implement with imu_features.py
        features["imu"] = {
            "tremor_power": 0.0,  # Placeholder
            "orientation_stability": 0.0,
        }
    
    return features

# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

async def main():
    # Initialize system
    init_system()
    
    # Add project root to path
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import queues from api.py
    from web.api import shared_queue, command_queue, broadcast
    
    # Create FSM
    fsm = SessionFSM()
    
    # Patient's disease type
    current_disease = Disease.TYPE_STROKE
    
    print("🔄 Main loop started\n")
    
    while True:
        # ═══════════════════════════════════════════════════════════
        # 1. CHECK USER COMMANDS
        # ═══════════════════════════════════════════════════════════
        
        try:
            cmd = command_queue.get_nowait()
            cmd_type = cmd.get("type", "unknown")
            
            print(f"📥 Command received: {cmd_type}")
            
            # ─────── LOGIN ───────
            if cmd_type == "login":
                fsm.context["patient_id"] = cmd.get("patient_id")
                fsm.context["patient_group"] = cmd.get("group", 1)
                
                # Determine disease type
                current_disease = Disease.TYPE_STROKE if cmd.get("group") == 1 else Disease.TYPE_TREMOR
                
                # Transition
                if fsm.transition("login", cmd):
                    await broadcast({
                        "type": "login_success",
                        "patient_id": fsm.context["patient_id"],
                        "state": fsm.state.value
                    })
            
            # ─────── START TRIAL ───────
            elif cmd_type == "start_trial":
                fsm.context["exercise"] = cmd.get("exercise", 1)
                fsm.context["trial_num"] = cmd.get("trial", 0)

                is_mvc = cmd.get("is_mvc", False)
    
                if is_mvc:
                    print("🔧 MVC Calibration recording...")
                    # Usar configuración especial para MVC
                    fsm.context["current_record_secs"] = cmd.get("record_secs", 5)
                    fsm.context["current_rest_secs"] = cmd.get("rest_secs", 0)  # Sin descanso
                else:
                    trial_num = fsm.context["trial_num"]
                    print(f"🎬 Trial {trial_num + 1} recording...")
                    # Usar configuración normal del ejercicio
                    # (ya deberían estar en fsm.context desde change_exercise)
                    fsm.context["current_record_secs"] = fsm.context.get("record_secs", 5)
                    fsm.context["current_rest_secs"] = fsm.context.get("rest_secs", 30)
                    
                    print(f"   📋 Config: current_record_secs={fsm.context['current_record_secs']}, current_rest_secs={fsm.context['current_rest_secs']}")

                # Transition
                if fsm.transition("start_trial", cmd):
                    await broadcast({
                        "type": "trial_started",
                        "trial": fsm.context["trial_num"],
                        "state": fsm.state.value
                    })
            
            # ─────── STOP TRIAL ───────
            elif cmd_type == "stop_trial":
                print("⏹️  Trial stopped manually - discarding data")
                
                # Descartar datos del trial actual
                fsm.context["trial_data"] = []
                
                if fsm.transition("stop_trial"):
                    await broadcast({
                        "type": "trial_discarded",
                        "trial": fsm.context["trial_num"],
                        "reason": "stopped_manually",
                        "state": fsm.state.value
                    })
        
            # ─────── CALIBRATE MVC ───────
            elif cmd_type == "calibrate_mvc":

                fsm.context["mvc"] = cmd.get("mvc_value", 100)
                print(f"🎯 MVC calibrated: {fsm.context['mvc']}")
            
            # ─────── CHANGE EXERCISE ───────
            elif cmd_type == "change_exercise":
                exercise_num = cmd.get("exercise", 1)
                
                fsm.context["exercise"] = exercise_num
                fsm.context["trial_num"] = 0
                fsm.context["record_secs"] = cmd.get("record_secs", 5)
                fsm.context["rest_secs"] = cmd.get("rest_secs", 30)
                fsm.context["total_trials"] = cmd.get("trials", 4)
                
                print(f"📋 Config: record={fsm.context['record_secs']}s, rest={fsm.context['rest_secs']}s")
                
                if fsm.transition("change_exercise"):
                    print(f"🔄 Changed to Exercise {exercise_num}")

            # ─────── RESET EXERCISE ───────
            elif cmd_type == "reset_exercise":
                # Resetear contadores
                fsm.context["trial_num"] = 0
                fsm.context["trial_data"] = []
                
                print(f"🔄 Exercise reset - back to start")
                
                if fsm.transition("reset_exercise"):
                    await broadcast({
                        "type": "exercise_reset",
                        "exercise": fsm.context.get("exercise", 1),
                        "state": fsm.state.value
                    })
        
        except asyncio.QueueEmpty:
            pass  # No commands
        
        # ═══════════════════════════════════════════════════════════
        # 2. READ SENSOR FRAMES
        # ═══════════════════════════════════════════════════════════
        
        try:
            frame = shared_queue.get_nowait()
            
            # If recording, accumulate
            if fsm.state == State.TRIAL_RECORDING:
                fsm.context["trial_data"].append(frame)
        
        except asyncio.QueueEmpty:
            pass  # No frames
        
        # ═══════════════════════════════════════════════════════════
        # 2.5. CHECK TIMEOUT (even if no frames)
        # ═══════════════════════════════════════════════════════════
        
        if fsm.state == State.TRIAL_RECORDING:
            elapsed = (datetime.now() - fsm.context["trial_start_time"]).total_seconds()
            duration = fsm.context.get("current_record_secs", 5)
            
            if elapsed >= duration:
                print(f"⏱️  Timeout reached ({duration}s) - {len(fsm.context['trial_data'])} frames captured")
                fsm.transition("timeout")
        
       
        # ═══════════════════════════════════════════════════════════
        # 3. PROCESS BY STATE
        # ═══════════════════════════════════════════════════════════
        
        if fsm.state == State.TRIAL_PROCESSING:
            # Extract features
            features = extract_trial_features(
                fsm.context["trial_data"],
                current_disease,
                fsm.context["mvc"]
            )
            
            # Send result to frontend
            await broadcast({
                "type": "trial_complete",
                "trial": fsm.context["trial_num"],
                "exercise": fsm.context["exercise"],
                "features": features,
                "state": fsm.state.value
            })
            
            # Transition to REST
            fsm.transition("features_ready")
        
        elif fsm.state == State.TRIAL_REST:
            # Check rest timeout (30 seconds)
            elapsed = (datetime.now() - fsm.context["rest_start_time"]).total_seconds()
            rest_duration = fsm.context.get("current_rest_secs", 30)            

            if elapsed >= rest_duration:
                # Increment trial
                fsm.context["trial_num"] += 1
                
                # More trials?
                if fsm.context["trial_num"] < fsm.context["total_trials"]:
                    fsm.transition("rest_complete")
                    print(f"✅ Rest finished. Ready for trial {fsm.context['trial_num'] + 1}")
                else:
                    fsm.transition("exercise_complete")
                    print("🏁 Exercise complete!")
                    await broadcast({
                        "type": "exercise_complete",
                        "exercise": fsm.context.get("exercise", 1),
                        "total_trials": fsm.context.get("total_trials", 4)
                    })
        
        # Small pause to not saturate CPU
        await asyncio.sleep(0.01)

# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")