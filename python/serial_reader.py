#function to read from the Arduino

# read_serial ()
#connect via Bluethooth, parse CSV and return the array
# serial_reader.py
import asyncio
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "SensorNode"
UART_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # notify

class BLEReader:
    def __init__(self):
        self.line_buffer = ""
        self.lines = asyncio.Queue()
        self.client = None

    async def connect(self):
        devices = await BleakScanner.discover()
        sensor = next((d for d in devices if d.name == DEVICE_NAME), None)
        if not sensor:
            raise Exception(f"{DEVICE_NAME} not found")

        self.client = BleakClient(sensor.address)
        await self.client.connect()
        await self.client.start_notify(UART_TX_UUID, self._handle_rx)
        print(f"Connected to {DEVICE_NAME}")

    def _handle_rx(self, sender, data: bytearray):
        text = data.decode(errors='ignore')
        for c in text:
            if c == '\n':
                asyncio.create_task(self.lines.put(self.line_buffer.strip()))
                self.line_buffer = ""
            elif c != '\r':
                self.line_buffer += c

    async def read_line(self):
        return await self.lines.get()