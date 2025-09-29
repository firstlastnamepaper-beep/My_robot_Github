import csv
import time
import os
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion
import rclpy
from rclpy.node import Node
from datetime import datetime
import atexit

class StateEstimator(Node):
    def __init__(self):
        super().__init__('state_estimator')
        self.pub = self.create_publisher(Float32MultiArray, 'mpc_state', 10)
        self.sub = self.create_subscription(
            PoseStamped,
            '/visual_slam/tracking/vo_pose',
            self.cb, 10)

        self.curr_pose = None
        self.prev_time = None
        self.prev_pos = None
        self.start_time = None
        self.log_data = []

        self.duration = 50.0  # seconds
        self._log_saved = False
        self.log_filename = self.get_timestamped_filename()
        self.create_timer(0.033, self.loop)

        # Ensure log gets saved even on Ctrl+C
        atexit.register(self.write_log)

    def cb(self, msg: PoseStamped):
        self.curr_pose = msg

    def loop(self):
        if self.curr_pose is None:
            return

        msg = self.curr_pose
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        q = msg.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = t

        vx = vy = vz = 0.0
        if self.prev_time is not None:
            dt = t - self.prev_time
            if dt > 0.001:
                vx = (x - self.prev_pos[0]) / dt
                vy = (y - self.prev_pos[1]) / dt
                vz = (z - self.prev_pos[2]) / dt

        self.prev_time = t
        self.prev_pos  = (x, y, z)
        gz = -9.81

        state = [roll, pitch, yaw,
                 x, y, z,
                 0.0, 0.0, 0.0,
                 vx, vy, vz,
                 gz]

        self.pub.publish(Float32MultiArray(data=state))

        # Log data
        elapsed = t - self.start_time
        self.log_data.append([elapsed] + state)

        # if elapsed >= self.duration:
        #     self.write_log()
        #     self.get_logger().info("50 seconds reached. Shutting down.")
        #     rclpy.shutdown()

    def write_log(self):
        if self._log_saved or not self.log_data:
            return
        self._log_saved = True

        with open(self.log_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['time'] + [f'state_{i}' for i in range(13)]
            writer.writerow(header)
            writer.writerows(self.log_data)

        self.get_logger().info(f"State log written to '{os.path.abspath(self.log_filename)}'")

    def get_timestamped_filename(self):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"robot_state_log_{now}.csv"

def main(args=None):
    rclpy.init(args=args)
    node = StateEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
