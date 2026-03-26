#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import math


class WallFollow(Node):

    def __init__(self):
        super().__init__('wall_follow_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.kp = 1.55
        self.kd = 0.55
        self.ki = 0.0


        self.desired_distance_right = 0.9
        self.lookahead_distance = 1.05

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = self.get_clock().now()
        

    def get_range(self, range_data, angle):

        scan = range_data

        #측정 될수 있는 angle 의 최대 각과 최소각 을 보정해 주는 코드
        angle = max(scan.angle_min, min(angle, scan.angle_max))
        #angle 에 대한 정보를 통해 현재 angle 에 최소각도를 뺴고 angle_increment 를 나누어서 index의 번호를 얻는 코드
        # (추후 이 index 번호를 range 데이터에 넣어 그 index의 번호의 range를 얻는 방법)
        i = int(round((angle - scan.angle_min) / scan.angle_increment))
        #index의 번호가 이상한 값이 들어올 경우 보정해주는 코드
        i = max(0, min(i, len(scan.ranges) - 1))

        #index의 번호를 넣어서 거리 r을 구하는 부분
        r = scan.ranges[i]  
        
        #r의 값이 없거나 유한하지 않은 경우 
        #scan.range_max if scan.range_max > 0.0 else 100.0를 반환하게 하는 코드
        if r is None or not math.isfinite(r) or r <= 0.0:
            return scan.range_max if scan.range_max > 0.0 else 100.0

        # range 의 최솟값보다 적고 최대값보다 넘치는 경우 이 값을 보정해주는 부분
        if scan.range_min > 0.0 and r < scan.range_min:
            r = scan.range_min
        if scan.range_max > 0.0 and r > scan.range_max:
            r = scan.range_max

        return float(r)

    def get_error(self, range_data, dist):
        
        #math를 이용해서 각도를 부여하는 코드
        angle_b = math.radians(-90.0)
        # if follow the left wall 
        # angle_b = math.radians(+90.0)  
        angle_a = math.radians(-45.0)  
        # if follow the left wall
        # angle_a = math.radians(+40.0)

        #부여받은 각도를 통해 range를 구하는 코드
        b = self.get_range(range_data, angle_b)
        a = self.get_range(range_data, angle_a)

        #두 각도의 차를 구해 세타를 구함
        theta = angle_a - angle_b  
        # if follow the left wall 
        # theta = angle_b - angle_a

        #-------------------------------------
        #교수님이 알려주신 공식들로 코드 구현
        k = a * math.sin(theta)
        if abs(k) < 1e-6:
            alpha = 0.0
        else:
            alpha = math.atan((a * math.cos(theta) - b) / k)

        D_t = b * math.cos(alpha)
        D_t_1 = D_t + self.lookahead_distance * math.sin(alpha)

        error = dist - D_t_1
        return float(error)

    def pid_control(self, error, velocity):
        
        #조건문을 통해 delta_time이 0이 되면 1e-3으로 반환
        current_time = self.get_clock().now()
        delta_time = (current_time - self.prev_time).nanoseconds / 1e9 
        if delta_time <= 0.0:
            delta_time = 1e-3

        #미분항 / 적분항 계산
        d_error = (error - self.prev_error) / delta_time
        self.integral += error * delta_time

        #PID 제어 공식에 따라 조향각 계산
        steering_angle = self.kp * error + self.ki * self.integral + self.kd * d_error

        #속도계산
        max_steer = math.radians(30.0)
        steering_angle = max(-max_steer, min(max_steer, steering_angle))
        # if follow the left wall => steering_angle = - max(-max_steer, min(max_steer, steering_angle))
        
        abs_steer = abs(steering_angle)
        if abs_steer > math.radians(20):
            velocity = 1.6
        elif abs_steer > math.radians(10):
            velocity = 2.5
        else:
            velocity = 3.4

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)

        self.prev_error = error
        self.prev_time = current_time

    def scan_callback(self, msg):
        error = self.get_error(msg, self.desired_distance_right)
        velocity = 1.0
        self.pid_control(error, velocity)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
