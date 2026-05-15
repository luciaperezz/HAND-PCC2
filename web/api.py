from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import os

app = FastAPI()

app.mount("/images", StaticFiles(directory="web/images"), name="images")


# Allow connections from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# SHARED QUEUES (for communication with main.py)
# ═══════════════════════════════════════════════════════════════

# shared_queue: BLE frames from browser → main.py
shared_queue = asyncio.Queue()

# command_queue: user commands → main.py
command_queue = asyncio.Queue()

# connected_clients: for broadcast
connected_clients = []

# ═══════════════════════════════════════════════════════════════
# BROADCAST FUNCTION (used by main.py to send results)
# ═══════════════════════════════════════════════════════════════

async def broadcast(data):
    """Sends data to ALL connected WebSocket clients"""
    dead = []
    for client in connected_clients:
        try:
            await client.send_json(data)
        except:
            dead.append(client)
    
    # Clean up dead clients
    for client in dead:
        if client in connected_clients:
            connected_clients.remove(client)

# ═══════════════════════════════════════════════════════════════
# STATIC ROUTES
# ═══════════════════════════════════════════════════════════════

# Serve the static folder at /static
# Get the directory where this api.py file is located
current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=current_dir, html=True), name="static")

# Route to serve your HTML at /
@app.get("/")
async def get_index():
    # Find index.html relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(current_dir, "index.html")
    
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>Error: index.html not found</h1><p>Expected at: {}</p>".format(index_path), status_code=500)
    
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# ═══════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("✅ WebSocket connected")
    connected_clients.append(ws)
    
    try:
        while True:
            # Receive message from browser
            msg = await ws.receive_text()
            
            # Try to parse as JSON
            try:
                data = json.loads(msg)
                msg_type = data.get("type", "unknown")
                
                #print(f"📥 Received from browser: {msg_type}")
                
                # ─────────────────────────────────────────────────
                # CLASSIFY MESSAGE
                # ─────────────────────────────────────────────────
                
                if msg_type == "sensor_frame":
                    # BLE frame parsed by JavaScript
                    # → Put in shared_queue for main.py
                    await shared_queue.put(data)
                
                elif msg_type in ["login", "start_trial", "stop_trial", "calibrate_mvc", "change_exercise","reset_exercise"]:
                    # User command (button clicked)
                    # → Put in command_queue for main.py
                    await command_queue.put(data)
                    print(f"📨 Command sent to main.py: {msg_type}")
                
                else:
                    # Other messages (ping, etc)
                    if msg == "ping":
                        await ws.send_text("pong")
            
            except json.JSONDecodeError:
                # Not JSON, handle as plain text
                if msg == "ping":
                    await ws.send_text("pong")
    
    except WebSocketDisconnect:
        if ws in connected_clients:
            connected_clients.remove(ws)
        print("❌ WebSocket disconnected")