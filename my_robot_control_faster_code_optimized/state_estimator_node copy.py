#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np
from tf_transformations import euler_from_quaternion

# class StateEstimator(Node):
#     def __init__(self):
#         super().__init__('state_estimator')
#         self.pub = self.create_publisher(Float32MultiArray, 'mpc_state', 10)
#         self.sub = self.create_subscription(
#             PoseStamped,
#             '/visual_slam/tracking/vo_pose',
#             self.cb, 10)
#         self.prev_time = None
#         self.prev_pos = None

#     def cb(self, msg: PoseStamped):
#         x = msg.pose.position.x
#         y = msg.pose.position.y
#         z = msg.pose.position.z
#         q = msg.pose.orientation
#         roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
#         t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

#         vx = vy = vz = 0.0
#         if self.prev_time is not None:
#             dt = t - self.prev_time
#             vx = (x - self.prev_pos[0]) / dt
#             vy = (y - self.prev_pos[1]) / dt
#             vz = (z - self.prev_pos[2]) / dt

#         self.prev_time = t
#         self.prev_pos  = (x, y, z)
#         gz = -9.81

#         state = [roll, pitch, yaw,
#                  x, y, z,
#                  0.0, 0.0, 0.0,  # angular vel (not used)
#                  vx, vy, vz,
#                  gz]
#         self.pub.publish(Float32MultiArray(data=state))


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

        self.create_timer(0.033, self.loop)  # 30Hz

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


def main(args=None):
    rclpy.init(args=args)
    node = StateEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()    

