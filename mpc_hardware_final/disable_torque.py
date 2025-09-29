#!/usr/bin/env python3
from dynamixel_sdk import PortHandler, PacketHandler

# adjust to your port and IDs
PORT_NAME      = '/dev/ttyTHS1'
BAUDRATE       = 1000000
PROTOCOL       = 2.0
TORQUE_ADDR    = 64
SERVO_IDS      = [1, 4, 2, 3]  # FL, FR, BL, BR

def main():
    port = PortHandler(PORT_NAME)
    if not port.openPort():
        print(f"ERROR: could not open {PORT_NAME}")
        return
    port.setBaudRate(BAUDRATE)
    packet = PacketHandler(PROTOCOL)

    for dxl_id in SERVO_IDS:
        result, err = packet.write1ByteTxRx(port, dxl_id, TORQUE_ADDR, 0)
        if result == packet.COMM_SUCCESS and err == 0:
            print(f"Torque disabled on ID {dxl_id}")
        else:
            print(f"Failed to disable torque on ID {dxl_id}: comm={result}, err={err}")

    port.closePort()

if __name__ == '__main__':
    main()
