#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class HolonomicDriveController(Node):
    def __init__(self):
        super().__init__('holonomic_drive_controller')
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.publisher = self.create_publisher(Float64MultiArray, '/wheel_velocities', 10)
        self.lx = 0.2
        self.ly = 0.2
        self.max_linear_vel = 1.0
        self.max_angular_vel = 1.5

    def cmd_vel_callback(self, msg):
        vx = max(min(msg.linear.x, self.max_linear_vel), -self.max_linear_vel)
        vy = max(min(msg.linear.y, self.max_linear_vel), -self.max_linear_vel)
        wz = max(min(msg.angular.z, self.max_angular_vel), -self.max_angular_vel)

        v_fl = vx - vy - wz * (self.lx + self.ly)
        v_fr = vx + vy + wz * (self.lx + self.ly)
        v_rl = vx + vy - wz * (self.lx + self.ly)
        v_rr = vx - vy + wz * (self.lx + self.ly)

        wheel_msg = Float64MultiArray()
        wheel_msg.data = [v_fl, v_fr, v_rl, v_rr]
        self.publisher.publish(wheel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HolonomicDriveController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
