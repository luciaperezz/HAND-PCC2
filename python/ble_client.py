import asyncio
from bleak import BleakClient, BleakScanner

DEVICE_NAME = "SensorNode"  # Debe coincidir con el nombre de la XIAO
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # Notify
UART_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # Write

async def main():
    # Escanea dispositivos
    print("Buscando dispositivos BLE...")
    devices = await BleakScanner.discover()
    sensor = next((d for d in devices if d.name == DEVICE_NAME), None)

    if not sensor:
        print(f"No se encontró {DEVICE_NAME}")
        return

    async with BleakClient(sensor.address) as client:
        print(f"Conectado a {DEVICE_NAME}")

        # Callback para recibir datos de TX
        def handle_rx(sender, data: bytearray):
            print("Recibido de XIAO:", data.decode().strip())

        # Suscribirse a TX (notify)
        await client.start_notify(UART_TX_UUID, handle_rx)
        print("Suscrito a TX (notify)")

        # Función para enviar datos al RX de la XIAO
        async def enviar_a_xiao(msg: str):
            await client.write_gatt_char(UART_RX_UUID, msg.encode())
            print("Enviado a XIAO:", msg)

        # Ejemplo: enviar datos cada 3 segundos
        contador = 0
        while True:
            await asyncio.sleep(3)
            mensaje = f"Hola XIAO! {contador}"
            await enviar_a_xiao(mensaje)
            contador += 1

if __name__ == "__main__":
    asyncio.run(main())