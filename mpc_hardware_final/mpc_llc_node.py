#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from my_robot_control.mpc_lib import mpc_control, LowLevelControl

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_llc_controller')
        self.sub = self.create_subscription(
            Float32MultiArray, 'mpc_state', self.state_cb, 10)
        self.pub = self.create_publisher(
            Float32MultiArray, 'theta_command', 10)

        self.state = None
        self.dt    = 0.033
        self.t     = 0.0
        self.llc   = LowLevelControl()
        self.init  = False
        self.create_timer(self.dt, self.loop)

    def state_cb(self, msg):
        arr = np.array(msg.data, dtype=float)
        if arr.shape[0] == 13:
            self.state = arr

    def loop(self):
        if self.state is None:
            return
        if not self.init:
            u0 = mpc_control(self.state, self.t)
            self.llc.pre_init(u0)
            self.init = True

        u_des = mpc_control(self.state, self.t)
        # get back the hardware‐mapped angles directly
        theta = self.llc.apply(u_des, self.t)
        self.pub.publish(Float32MultiArray(data=theta.tolist()))
        self.t += self.dt

def main(args=None):
    rclpy.init(args=args)
    node = MPCController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

