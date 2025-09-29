#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite

class MotorCommand(Node):
    def __init__(self):
        super().__init__('motor_command')
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        port_name = self.get_parameter('serial_port') \
                        .get_parameter_value().string_value
        self.get_logger().info(f'Using serial port: {port_name}')

        # ---- Dynamixel setup
        self.port = PortHandler(port_name)
        if not self.port.openPort():
            self.get_logger().error(f'Failed to open port {port_name}')
            raise RuntimeError("Could not open Dynamixel serial port")
        self.port.setBaudRate(2000000)
        self.packet = PacketHandler(2.0)

        # GroupSyncWrite: addr=116 (Goal Position, X-series), length=4 bytes
        self.group = GroupSyncWrite(self.port, self.packet, 116, 4)

        # FL→1, FR→4, BL→2, BR→3  (your mapping)
        self.ids = [2, 3, 5, 4]

        for dxl_id in self.ids:
            # Torque Enable (address 64) = 1
            dxl_comm_result, dxl_error = self.packet.write1ByteTxRx(self.port, dxl_id, 64, 1)
            if dxl_comm_result != 0 or dxl_error != 0:
                self.get_logger().warn(f"Torque enable failed for ID {dxl_id} "
                                       f"(comm={dxl_comm_result}, err={dxl_error})")
        self.get_logger().info("All servos torque-enabled")

        # ---- Timing / metrics
        self.last_arrival = None
        self.hz_samples = []          # instantaneous Hz samples
        self.cb_time_samples = []     # callback wall-time per message (ms)
        self.tx_time_samples = []     # txPacket() time (ms)

        # Print a brief metrics line every second
        self.metrics_timer = self.create_timer(1.0, self._print_metrics)

        # ---- Subscriber
        self.sub = self.create_subscription(
            Float32MultiArray, 'theta_command', self.cb, 10)

    def cb(self, msg: Float32MultiArray):
        t0 = time.perf_counter()

        # Inter-arrival timing & instantaneous Hz
        now = t0
        if self.last_arrival is not None:
            dt = now - self.last_arrival
            if dt > 0:
                self.hz_samples.append(1.0 / dt)
        self.last_arrival = now

        thetas = msg.data  # [θ_FL, θ_FR, θ_BL, θ_BR] in degrees
        self.get_logger().info(f'Commanding angles (deg): {["{:.1f}".format(t) for t in thetas]}')

        # Convert degrees → raw counts (0–4095 for 0–360°)
        counts = [int(t/360.0*4096) for t in thetas]

        # Prepare and send the sync‐write
        self.group.clearParam()
        for dxl_id, cnt in zip(self.ids, counts):
            param = [
                cnt & 0xFF,
                (cnt >> 8) & 0xFF,
                (cnt >> 16) & 0xFF,
                (cnt >> 24) & 0xFF,
            ]
            ok = self.group.addParam(dxl_id, param)
            if not ok:
                self.get_logger().warn(f"addParam failed for ID {dxl_id}")

        t_tx0 = time.perf_counter()
        dxl_comm_result = self.group.txPacket()  # returns 0 on success
        t_tx1 = time.perf_counter()

        # Record tx time
        self.tx_time_samples.append((t_tx1 - t_tx0) * 1000.0)

        if dxl_comm_result != 0:
            self.get_logger().warn(f"txPacket() returned non-zero: {dxl_comm_result}")

        # Total callback time
        t1 = time.perf_counter()
        self.cb_time_samples.append((t1 - t0) * 1000.0)

    def _print_metrics(self):
        def avg(lst):
            return sum(lst)/len(lst) if lst else 0.0

        avg_hz   = avg(self.hz_samples)
        avg_cbms = avg(self.cb_time_samples)
        avg_txms = avg(self.tx_time_samples)

        # Print concise one-line status
        self.get_logger().info(
            f"[timing] incoming ~{avg_hz:6.2f} Hz | cb ~{avg_cbms:6.2f} ms | txPacket ~{avg_txms:6.2f} ms "
            f"(n={len(self.cb_time_samples)})"
        )

        # Keep windows bounded so numbers stay “recent”
        max_keep = 300  # ~ last few seconds depending on rate
        self.hz_samples       = self.hz_samples[-max_keep:]
        self.cb_time_samples  = self.cb_time_samples[-max_keep:]
        self.tx_time_samples  = self.tx_time_samples[-max_keep:]

def main(args=None):
    rclpy.init(args=args)
    node = MotorCommand()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
