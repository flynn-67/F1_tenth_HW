import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import math


class WallFollow(Node):
    """
    목표:
    1) 직진 구간에서는 오른쪽 벽에 붙어서 최대한 빠르게 전진
    2) 오른쪽에 갑자기 열리는 함정(옆으로 파인 공간)에 빨려 들어가지 않기
    3) 코너에서는 한 방향(왼쪽 회전)으로 확실하게 감속 후 통과
    """

    def __init__(self):
        super().__init__('wall_follow_node')

        # 토픽
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10
        )

        # =========================
        # 기본 설정
        # =========================

        # 조향 방향이 반대로 먹으면 1.0 -> -1.0 으로 바꾸면 됨
        self.steering_sign = 1.0

        # 직진용 PID (오른쪽 벽 추종)
        self.straight_kp = 0.85
        self.straight_kd = 0.14

        # 코너용 PID (왼쪽 벽 추종)
        self.corner_kp = 1.10
        self.corner_kd = 0.18

        self.ki = 0.0

        # 목표 벽 거리
        self.desired_distance_right = 0.75
        self.desired_distance_left = 0.75

        # lookahead
        # 직진은 조금 멀리 보고, 코너는 짧게 봐서 급발진/과회전 줄임
        self.right_lookahead = 0.90
        self.left_lookahead = 0.65

        # 최대 조향각 제한
        self.max_steering = math.radians(24.0)

        # =========================
        # 속도 설정
        # =========================
        self.straight_speed_max = 5.5   # 직진 최고속도
        self.straight_speed_mid = 4.6
        self.straight_speed_low = 3.8

        self.trap_guard_speed = 4.2     # 함정 회피 중 속도
        self.corner_speed = 1.8         # 일반 코너
        self.hairpin_speed = 1.1        # 매우 급한 코너

        # =========================
        # 코너 / 함정 판단 기준
        # =========================

        # 앞이 이 정도보다 막히면 코너로 보기 시작
        self.corner_front_threshold = 1.80
        self.hairpin_front_threshold = 1.05

        # 오른쪽이 갑자기 크게 열리면 함정으로 판단
        self.trap_right_open_threshold = 2.60
        self.trap_right_front_open_threshold = 2.20

        # 함정 감지 후 몇 프레임 동안 오른쪽 벽을 무시하고 버틸지
        self.trap_hold_frames = 12
        self.trap_hold_count = 0

        # 함정 구간에서 살짝 왼쪽으로 버티는 조향
        self.trap_guard_steer = math.radians(6.0)

        # PID 상태값
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = self.get_clock().now()

        # 직전 정상적인 오른쪽 벽 추종 오차 저장
        self.last_safe_right_error = 0.0

        # 로그용
        self.log_count = 0
        self.mode = "INIT"

    def get_range(self, scan_msg, angle):
        """
        특정 각도(rad)에 해당하는 라이다 거리값을 안전하게 가져온다.
        NaN, inf, 범위 밖 값은 큰 값으로 처리
        """
        if angle < scan_msg.angle_min or angle > scan_msg.angle_max:
            return 100.0

        index = int((angle - scan_msg.angle_min) / scan_msg.angle_increment)
        index = max(0, min(index, len(scan_msg.ranges) - 1))

        distance = scan_msg.ranges[index]

        if math.isnan(distance) or math.isinf(distance):
            return 100.0

        return distance

    def get_wall_error(self, scan_msg, dist, follow_left=False, lookahead=0.7):
        """
        벽과의 상대 각도를 이용해서 wall-follow 오차 계산
        follow_left=False -> 오른쪽 벽 기준
        follow_left=True  -> 왼쪽 벽 기준
        """
        theta = math.radians(45)

        if follow_left:
            angle_b = math.pi / 2
            angle_a = angle_b - theta
        else:
            angle_b = -math.pi / 2
            angle_a = angle_b + theta

        b = self.get_range(scan_msg, angle_b)
        a = self.get_range(scan_msg, angle_a)

        denominator = a * math.sin(theta)
        if abs(denominator) < 1e-6:
            return 0.0

        alpha = math.atan((a * math.cos(theta) - b) / denominator)
        current_dist = b * math.cos(alpha)
        future_dist = current_dist + lookahead * math.sin(alpha)

        if follow_left:
            # 왼쪽 벽 추종
            error = future_dist - dist
        else:
            # 오른쪽 벽 추종
            error = dist - future_dist

        return error

    def publish_drive(self, steering_angle, speed):
        """
        최종 조향각과 속도를 차량에 publish
        """
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = self.steering_sign * steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def pid_control(self, error, base_speed, mode, front_distance):
        """
        모드에 따라 다른 PID를 적용하고,
        조향각 크기 / 전방 거리 기준으로 최종 속도를 다시 조절
        """
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 0.001

        derivative = (error - self.prev_error) / dt
        self.integral += error * dt

        # 적분항은 이번 문제에서는 크게 필요 없어서 제한만 걸고 사실상 0처럼 사용
        self.integral = max(-1.0, min(1.0, self.integral))

        if mode == "CORNER":
            kp = self.corner_kp
            kd = self.corner_kd
        else:
            kp = self.straight_kp
            kd = self.straight_kd

        steering_angle = (
            kp * error
            + self.ki * self.integral
            + kd * derivative
        )

        # 최대 조향각 제한
        steering_angle = max(-self.max_steering, min(self.max_steering, steering_angle))

        # 기본 속도에서 추가 감속
        speed = base_speed
        abs_deg = abs(math.degrees(steering_angle))

        if mode == "STRAIGHT":
            if abs_deg > 14.0:
                speed *= 0.78
            elif abs_deg > 8.0:
                speed *= 0.90

        elif mode == "CORNER":
            if abs_deg > 18.0:
                speed *= 0.72
            elif abs_deg > 10.0:
                speed *= 0.84

        # 전방이 갑자기 막히면 추가 감속
        if front_distance < 0.95:
            speed = min(speed, 0.9)
        elif front_distance < 1.30:
            speed = min(speed, 1.2)
        elif front_distance < 1.80:
            speed = min(speed, 1.7)

        self.publish_drive(steering_angle, speed)

        self.prev_error = error
        self.prev_time = now

        self.log_count += 1
        if self.log_count % 10 == 0:
            self.get_logger().info(
                f"mode={mode}, "
                f"front={front_distance:.2f}, "
                f"error={error:.3f}, "
                f"steer_deg={math.degrees(steering_angle):.2f}, "
                f"speed={speed:.2f}, "
                f"trap_hold={self.trap_hold_count}"
            )

    def detect_right_trap(self, front, right, right_front, left_front):
        """
        오른쪽에 파인 함정 / 옆으로 열린 공간 감지
        핵심 아이디어:
        - 앞은 아직 충분히 뚫려 있음
        - 오른쪽 / 오른쪽 앞이 갑자기 매우 커짐
        - 즉, 오른쪽 벽이 사라진 것처럼 보이는 상황
        -> 이때 오른쪽 벽을 계속 따라가면 함정 안으로 빨려 들어갈 수 있음
        """
        right_suddenly_open = (
            right > self.trap_right_open_threshold and
            right_front > self.trap_right_front_open_threshold and
            front > 2.0 and
            left_front > 1.0
        )
        return right_suddenly_open

    def compute_straight_speed(self, front_distance):
        """
        직진 구간 속도 결정
        """
        if front_distance > 6.0:
            return self.straight_speed_max
        elif front_distance > 4.0:
            return self.straight_speed_mid
        else:
            return self.straight_speed_low

    def scan_callback(self, msg):
        """
        메인 로직
        1) 직진: 오른쪽 벽 따라 빠르게 감
        2) 함정 감지: 잠깐 오른쪽 벽을 무시하고 살짝 왼쪽으로 버팀
        3) 코너: 왼쪽 벽 기준으로 power corner
        """
        # =========================
        # 주요 라이다 샘플
        # =========================
        front = self.get_range(msg, 0.0)
        front_left = self.get_range(msg, math.radians(35))
        front_right = self.get_range(msg, math.radians(-35))
        left_far = self.get_range(msg, math.radians(55))
        right_far = self.get_range(msg, math.radians(-55))
        left_side = self.get_range(msg, math.pi / 2)
        right_side = self.get_range(msg, -math.pi / 2)

        # 벽 추종 오차 계산
        right_error = self.get_wall_error(
            msg,
            self.desired_distance_right,
            follow_left=False,
            lookahead=self.right_lookahead
        )
        left_error = self.get_wall_error(
            msg,
            self.desired_distance_left,
            follow_left=True,
            lookahead=self.left_lookahead
        )

        # =========================
        # 1. 함정 감지
        # =========================
        trap_detected = self.detect_right_trap(
            front=front,
            right=right_side,
            right_front=right_far,
            left_front=left_far
        )

        if trap_detected:
            self.trap_hold_count = self.trap_hold_frames

        # 정상 직진 중일 때만 오른쪽 벽 기준 오차 저장
        if not trap_detected and right_side < self.trap_right_open_threshold:
            self.last_safe_right_error = right_error

        # =========================
        # 2. 함정 회피 우선
        # =========================
        if self.trap_hold_count > 0 and front > self.corner_front_threshold:
            self.mode = "TRAP_GUARD"

            # 함정 구간에서는 오른쪽 벽이 사라지므로
            # 오른쪽 wall-follow를 잠깐 버리고 살짝 왼쪽으로 버틴다.
            steer = self.trap_guard_steer

            # 앞이 엄청 뚫려 있으면 빠르게, 아니면 조금 낮춤
            speed = self.trap_guard_speed
            if front < 3.0:
                speed = min(speed, 3.4)

            self.publish_drive(steer, speed)
            self.trap_hold_count -= 1

            self.log_count += 1
            if self.log_count % 10 == 0:
                self.get_logger().info(
                    f"mode=TRAP_GUARD, front={front:.2f}, right={right_side:.2f}, "
                    f"right_front={right_far:.2f}, speed={speed:.2f}, "
                    f"steer_deg={math.degrees(steer):.2f}, trap_hold={self.trap_hold_count}"
                )
            return

        # =========================
        # 3. 코너 판단
        # =========================
        # 앞이 좁아지면 코너
        # 특히 좌회전만 주로 한다고 가정해서 front가 짧아지면 강하게 감속
        is_hairpin = front < self.hairpin_front_threshold
        is_corner = front < self.corner_front_threshold

        if is_hairpin:
            self.mode = "CORNER"
            base_speed = self.hairpin_speed
            self.pid_control(left_error, base_speed, "CORNER", front)
            return

        if is_corner:
            self.mode = "CORNER"
            base_speed = self.corner_speed
            self.pid_control(left_error, base_speed, "CORNER", front)
            return

        # =========================
        # 4. 직진
        # =========================
        self.mode = "STRAIGHT"
        base_speed = self.compute_straight_speed(front)
        self.pid_control(right_error, base_speed, "STRAIGHT", front)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollow()
    print("Right-wall fast + trap-guard + power-corner 노드 시작")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()