"""
API estable BLE + WebSocket
pip install fastapi uvicorn bleak
python api.py
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import asyncio
import logging
from datetime import datetime

from bleak import BleakClient, BleakScanner

# ─────────────────────────────
# LOGGING
# ─────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HAND_API")

# ─────────────────────────────
# APP
# ─────────────────────────────
app = FastAPI(title="Sensor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────
# GLOBAL STATE
# ─────────────────────────────
active_websockets = []
ble_bridge = None
last_frame = None
stats = {"frames": 0, "errors": 0}

# ─────────────────────────────
# WEBSOCKETS
# ─────────────────────────────
async def add_ws(ws):
    active_websockets.append(ws)
    logger.info(f"WS conectado ({len(active_websockets)})")

async def remove_ws(ws):
    if ws in active_websockets:
        active_websockets.remove(ws)
    logger.info(f"WS desconectado ({len(active_websockets)})")

async def broadcast(data):
    global last_frame, stats

    last_frame = data
    stats["frames"] += 1

    dead = []
    for ws in active_websockets:
        try:
            await ws.send_json(data)
        except:
            dead.append(ws)

    for ws in dead:
        await remove_ws(ws)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await add_ws(ws)

    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        await remove_ws(ws)
    except Exception:
        await remove_ws(ws)

# ─────────────────────────────
# BLE BRIDGE
# ─────────────────────────────
class BLEBridge:
    def __init__(self, name="SensorNode"):
        self.name = name
        self.client = None
        self.connected = False
        self.running = True

    async def find_device(self):
        logger.info("🔍 Escaneando BLE...")
        devices = await BleakScanner.discover(timeout=5)

        for d in devices:
            if d.name and self.name in d.name:
                logger.info(f"✅ Encontrado: {d.name}")
                return d.address

        return None

    async def handle(self, sender, data):
        try:
            msg = data.decode().strip()

            if not msg.startswith("F,"):
                return

            p = msg.split(",")

            if len(p) < 10:
                return

            frame = {
                "type": "sensor_frame",
                "timestamp": datetime.now().isoformat(),
                "frame_id": int(p[1]),
                "pressure": float(p[2]),
                "emg_raw": float(p[3]),
                "ax": float(p[4]),
                "ay": float(p[5]),
                "az": float(p[6]),
                "gx": float(p[7]),
                "gy": float(p[8]),
                "gz": float(p[9]),
            }

            await broadcast(frame)

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Parse error: {e}")

    async def connect(self):
        address = await self.find_device()
        if not address:
            logger.warning("❌ BLE no encontrado")
            return False

        self.client = BleakClient(address)
        await self.client.connect()

        self.connected = True
        logger.info("🔗 BLE conectado")

        # UART estándar Nordic
        UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

        try:
            await self.client.start_notify(UUID, self.handle)
            logger.info("📡 Notificaciones activadas")
            return True
        except Exception as e:
            logger.error(f"Notify error: {e}")
            return False

    async def loop(self):
        while self.running:
            if not self.connected:
                try:
                    ok = await self.connect()
                    if not ok:
                        await asyncio.sleep(3)
                except Exception as e:
                    logger.error(e)
                    await asyncio.sleep(3)
            else:
                await asyncio.sleep(1)

# ─────────────────────────────
# ROUTES
# ─────────────────────────────
@app.get("/")
async def home():
    return HTMLResponse("<h2>Sensor API running</h2>")

@app.get("/api/status")
async def status():
    return {
        "ws_clients": len(active_websockets),
        "frames": stats["frames"],
        "errors": stats["errors"],
        "last_frame": last_frame,
        "ble_connected": ble_bridge.connected if ble_bridge else False
    }

# ─────────────────────────────
# STARTUP
# ─────────────────────────────
@app.on_event("startup")
async def startup():
    global ble_bridge

    logger.info("🚀 Starting API...")

    ble_bridge = BLEBridge("SensorNode")

    asyncio.create_task(ble_bridge.loop())

    logger.info("✅ Ready")

# ─────────────────────────────
# RUN
# ─────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)