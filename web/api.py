from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Allow connections from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients = []  # ← NUEVO

# Serve the static folder at /static
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Route to serve your HTML at /
@app.get("/")
async def get_index():
    index_path = os.path.join(".", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("WebSocket connected ✅")
    connected_clients.append(ws)
    try:
        while True:
            data = await ws.receive_json()
            print("Received data:", data)
            response = {"result": "ok", "data": data}
            for client in connected_clients[:]:
                try:
                    await client.send_json(response)
                except:
                    connected_clients.remove(client)
    except WebSocketDisconnect:
        connected_clients.remove(ws)
        print("WebSocket disconnected")
