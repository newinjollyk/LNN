#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import os
import csv
from datetime import datetime
import numpy as np
import math

# ===================== USER SETTINGS (edit before each run) =====================
START_ID   = 'Home'   # 'Home' or 'A'/'B'/'C' if you're doing Goal->Home runs
GOAL_ID    = 'A'      # 'A' / 'B' / 'C' / 'Home'
EPISODE_ID = 5        # increment per run

# World/map frame coordinates from Ignition Gazebo 
GOALS = {
    'Home': {'x': 13.900000,   'y': -10.600000,  'yaw_deg': -0.000003},
    'A':    {'x': -13.638600,  'y': 2.687560, 'yaw_deg': -3.115870},
    'B':    {'x': -13.630200,  'y': 10.011400 , 'yaw_deg': -3.113550},
    'C':    {'x': -13.636200, 'y': 17.522800, 'yaw_deg': -3.114620},
}
# ================================================================================
def wrap_to_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def relative_goal(robot_pose, goal_pose):
    # robot_pose = (x_r, y_r, yaw_r) in world
    # goal_pose  = (x_g, y_g, yaw_g) in world
    x_r, y_r, th_r = robot_pose
    x_g, y_g, th_g = goal_pose
    dx = x_g - x_r
    dy = y_g - y_r
    c, s = math.cos(th_r), math.sin(th_r)
    dX =  c*dx + s*dy
    dY = -s*dx + c*dy
    dYaw = wrap_to_pi(th_g - th_r)
    return dX, dY, math.sin(dYaw), math.cos(dYaw)

def dist2d(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

class TugbotRecorder(Node):
    def __init__(self):
        super().__init__('tugbot_recorder')

        # ===== Goal selection & sanity =====
        assert START_ID in GOALS and GOAL_ID in GOALS, "START_ID/GOAL_ID must be keys in GOALS"
        self.start_id   = START_ID
        self.goal_id    = GOAL_ID
        self.episode_id = int(EPISODE_ID)

        # ===== Runtime caches =====
        self.current_image = None
        self.odom_linear_vel = 0.0
        self.odom_angular_vel = 0.0
        self.cmd_linear_vel = 0.0
        self.cmd_angular_vel = 0.0
        self.current_lidar = []
        self.detected_markers = []
        # robot pose cache
        self.x_r = 0.0
        self.y_r = 0.0
        self.yaw_r = 0.0

        # ===== File setup =====
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"dataset/{run_stamp}_{self.start_id}2{self.goal_id}_ep{self.episode_id}"
        img_folder = os.path.join(folder, "images")
        os.makedirs(img_folder, exist_ok=True)
        self.image_dir = img_folder

        csv_name = f"data_{self.start_id}2{self.goal_id}_ep{self.episode_id}.csv"
        self.csv_path = os.path.join(folder, csv_name)
        os.makedirs(folder, exist_ok=True)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # CSV header (includes episode_id & goal_id)
        self.csv_writer.writerow([
            'episode_id',
            'start_id','goal_id',
            'image_file',
            # robot & goal poses (world)
            'x_r','y_r','yaw_r',
            'x_g','y_g','yaw_g',
            # relative goal features (robot frame)
            'dX','dY','sin_dYaw','cos_dYaw',
            # distances
            'dist_to_goal','dist_home_to_goal',
            # velocities
            'odom_linear_vel','odom_angular_vel',
            'cmd_linear_vel','cmd_angular_vel',
            # sensors
            'lidar_points',
            # top 3 ArUco markers (id, x, y, z, yaw)
            'marker1_id','marker1_x','marker1_y','marker1_z','marker1_yaw',
            'marker2_id','marker2_x','marker2_y','marker2_z','marker2_yaw',
            'marker3_id','marker3_x','marker3_y','marker3_z','marker3_yaw'
        ])

        # ===== ArUco setup =====
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size = 0.6  # meters

        self.camera_matrix = np.array([
            [615.96, 0.0,   419.83],
            [0.0,   616.22, 245.14],
            [0.0,   0.0,    1.0   ]
        ])
        self.dist_coeffs = np.zeros((5, 1))

        # ===== ROS setup =====
        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            '/world/world_demo/model/tugbot/link/camera_front/sensor/color/image',
            self.image_callback, 10
        )
        self.create_subscription(
            Odometry,
            '/model/tugbot/odometry',
            self.odom_callback, 10
        )
        self.create_subscription(
            LaserScan,
            '/world/world_demo/model/tugbot/link/scan_front/sensor/scan_front/scan',
            self.lidar_callback, 10
        )
        self.create_subscription(
            Twist,
            '/model/tugbot/cmd_vel',
            self.cmd_callback, 10
        )

        # Timer at 5 Hz (0.2 s)
        self.timer = self.create_timer(0.2, self.save_data)

        # Precompute Home→Goal distance (static sanity)
        self.home_xy = (GOALS['Home']['x'], GOALS['Home']['y'])
        gx, gy = GOALS[self.goal_id]['x'], GOALS[self.goal_id]['y']
        self.dist_home_to_goal_static = dist2d(self.home_xy, (gx, gy))

        self.get_logger().info(f"Logging to: {self.csv_path}")
        self.get_logger().info(f"Episode {self.episode_id}: {self.start_id} → {self.goal_id} | Home→{self.goal_id} = {self.dist_home_to_goal_static:.3f} m")

    # --- Callbacks ---
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            self.detected_markers = []
            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                for i, marker_id in enumerate(ids.flatten()):
                    x, y, z = tvecs[i][0]
                    yaw = float(rvecs[i][0][2])  # simplified yaw (rotation around z in camera frame)
                    self.detected_markers.append((int(marker_id), float(x), float(y), float(z), float(yaw)))
                # sort by range (z) — closest first
                self.detected_markers.sort(key=lambda m: m[3])
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def odom_callback(self, msg):
        # velocities
        self.odom_linear_vel  = float(msg.twist.twist.linear.x)
        self.odom_angular_vel = float(msg.twist.twist.angular.z)
        # pose (assuming odom/world-aligned; otherwise use TF map->base_link)
        self.x_r = float(msg.pose.pose.position.x)
        self.y_r = float(msg.pose.pose.position.y)
        self.yaw_r = quat_to_yaw(msg.pose.pose.orientation)

    def cmd_callback(self, msg):
        self.cmd_linear_vel  = float(msg.linear.x)
        self.cmd_angular_vel = float(msg.angular.z)

    def lidar_callback(self, msg):
        # downsample every 10th beam for compact CSV
        self.current_lidar = [round(r, 3) for r in msg.ranges[::10]]

    # --- Save data at fixed rate ---
    def save_data(self):
        if self.current_image is None:
            return

        # goal (world frame)
        g = GOALS[self.goal_id]
        x_g, y_g = float(g['x']), float(g['y'])
        yaw_g = math.radians(float(g['yaw_deg']))

        # relative goal features (robot frame)
        dX, dY, sin_dYaw, cos_dYaw = relative_goal(
            (self.x_r, self.y_r, self.yaw_r),
            (x_g, y_g, yaw_g)
        )

        # distances
        dist_to_goal = dist2d((self.x_r, self.y_r), (x_g, y_g))
        dist_home_to_goal = self.dist_home_to_goal_static  # static sanity field

        # Save image
        img_name = f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
        cv2.imwrite(os.path.join(self.image_dir, img_name), self.current_image)

        # Prepare marker data (up to 3)
        markers_out = []
        for i in range(3):
            if i < len(self.detected_markers):
                markers_out.extend(self.detected_markers[i])
            else:
                markers_out.extend([None, None, None, None, None])

        # Write CSV row
        self.csv_writer.writerow([
            self.episode_id,
            self.start_id, self.goal_id,
            img_name,
            f"{self.x_r:.6f}", f"{self.y_r:.6f}", f"{self.yaw_r:.6f}",
            f"{x_g:.6f}", f"{y_g:.6f}", f"{yaw_g:.6f}",
            f"{dX:.6f}", f"{dY:.6f}", f"{sin_dYaw:.6f}", f"{cos_dYaw:.6f}",
            f"{dist_to_goal:.6f}", f"{dist_home_to_goal:.6f}",
            f"{self.odom_linear_vel:.6f}", f"{self.odom_angular_vel:.6f}",
            f"{self.cmd_linear_vel:.6f}", f"{self.cmd_angular_vel:.6f}",
            self.current_lidar,
            *markers_out
        ])

        self.get_logger().info(
            f"Saved {img_name} | ep={self.episode_id} {self.start_id}->{self.goal_id} "
            f"| d_goal={dist_to_goal:.2f} m | v={self.odom_linear_vel:.2f} m/s"
        )

    def destroy_node(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TugbotRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
