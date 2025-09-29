#!/usr/bin/env python3
from dynamixel_sdk import PortHandler, PacketHandler

def scan(port_name, protocol=2.0):
    print(f"\n--- Scanning {port_name} (Prot {protocol}) ---")
    port = PortHandler(port_name)
    if not port.openPort():
        print(f"  ✖ Cannot open port {port_name}")
        return
    port.setBaudRate(1000000)
    packet = PacketHandler(protocol)
    found = []
    for dxl_id in range(1, 11):
        model, comm, err = packet.ping(port, dxl_id)
        if comm == packet.COMM_SUCCESS and err == 0:
            print(f"  ✔ ID {dxl_id} → model {model}")
            found.append(dxl_id)
    if not found:
        print("  – no servos found")
    port.closePort()

if __name__=='__main__':
    for p in ['/dev/ttyTHS1','/dev/ttyTHS4']:
        scan(p, protocol=2.0)
