#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool


class SafetyNode(Node):

    def __init__(self):
        super().__init__('safety_node')

        self.brake_publisher = self.create_publisher(Bool, '/brake_bool', 10)
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.odom_subscriber = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10
        )

        self.lidar_data = None
        self.speed = 0.0

        # 필요하면 조절
        self.ttc_threshold = 0.7   # 초
        self.timer = self.create_timer(0.01, self.check_ttc)

    def odom_callback(self, odom_msg):
        # ego vehicle longitudinal speed
        # F1TENTH에서는 일반적으로 x 방향 선속도를 사용
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):
        self.lidar_data = scan_msg

    def check_ttc(self):
        if self.lidar_data is None:
            return

        # 뒤로 가는 중이거나 정지 중이면 TTC 계산 의미가 약함
        if self.speed <= 0.0:
            self.publish_brake(False)
            return

        ranges = np.array(self.lidar_data.ranges, dtype=np.float32)

        # angle 배열 생성
        angles = self.lidar_data.angle_min + np.arange(len(ranges)) * self.lidar_data.angle_increment

        # 각 빔 방향에서 전방 속도 성분
        range_rate = self.speed * np.cos(angles)

        # 전방으로 진행하는 빔만 사용
        valid_forward = range_rate > 0.0

        # inf / nan 제거
        valid_range = np.isfinite(ranges)

        valid = valid_forward & valid_range

        if not np.any(valid):
            self.publish_brake(False)
            return

        valid_ranges = ranges[valid]
        valid_rates = range_rate[valid]

        # TTC = distance / closing speed
        ttc = valid_ranges / valid_rates

        # 혹시 모를 이상값 제거
        ttc = ttc[np.isfinite(ttc)]

        if len(ttc) == 0:
            self.publish_brake(False)
            return

        min_ttc = np.min(ttc)

        # 디버깅용 로그
        # self.get_logger().info(f"speed={self.speed:.2f}, min_ttc={min_ttc:.2f}")

        if min_ttc < self.ttc_threshold:
            self.publish_stop_drive()
            self.publish_brake(True)
            self.get_logger().warn(f'BRAKE! min_ttc = {min_ttc:.3f}')
        else:
            self.publish_brake(False)

    def publish_brake(self, should_brake):
        brake_msg = Bool()
        brake_msg.data = should_brake
        self.brake_publisher.publish(brake_msg)

    def publish_stop_drive(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_publisher.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    print("Run safety_node")
    rclpy.spin(safety_node)

    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()