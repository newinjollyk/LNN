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

class TugbotRecorder(Node):
    def __init__(self):
        super().__init__('tugbot_recorder')

        # ===== Initialization =====
        self.current_image = None
        self.odom_linear_vel = 0.0
        self.odom_angular_vel = 0.0
        self.cmd_linear_vel = 0.0
        self.cmd_angular_vel = 0.0
        self.current_lidar = []
        self.detected_markers = []

        # ===== File setup =====
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"dataset/{timestamp}/images", exist_ok=True)
        self.image_dir = f"dataset/{timestamp}/images"

        self.csv_file = open(f"dataset/{timestamp}/data.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'image_file',
            'odom_linear_vel', 'odom_angular_vel',
            'cmd_linear_vel', 'cmd_angular_vel',
            'lidar_points',
            'marker1_id', 'marker1_x', 'marker1_y', 'marker1_z', 'marker1_yaw',
            'marker2_id', 'marker2_x', 'marker2_y', 'marker2_z', 'marker2_yaw',
            'marker3_id', 'marker3_x', 'marker3_y', 'marker3_z', 'marker3_yaw'
        ])

        # ===== ArUco setup =====
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size = 0.6  # marker size in meters (600 mm)


        self.camera_matrix = np.array([
            [615.96, 0.0,   419.83],
            [0.0,   616.22, 245.14],
            [0.0,   0.0,    1.0   ]
        ])

        self.dist_coeffs = np.zeros((5, 1)) 

        # ===== ROS setup =====
        self.bridge = CvBridge()

        # Subscriptions
        self.create_subscription(Image,
                                 '/world/world_demo/model/tugbot/link/camera_front/sensor/color/image',
                                 self.image_callback, 10)
        self.create_subscription(Odometry,
                                 '/model/tugbot/odometry',
                                 self.odom_callback, 10)
        self.create_subscription(LaserScan,
                                 '/world/world_demo/model/tugbot/link/scan_front/sensor/scan_front/scan',
                                 self.lidar_callback, 10)
        self.create_subscription(Twist,
                                 '/model/tugbot/cmd_vel',
                                 self.cmd_callback, 10)

        # Timer at 5Hz (every 0.2s)
        self.timer = self.create_timer(0.2, self.save_data)

    # --- Callbacks ---
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Detect markers
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            self.detected_markers = []
            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                for i, marker_id in enumerate(ids.flatten()):
                    x, y, z = tvecs[i][0]
                    yaw = float(rvecs[i][0][2])  # simplified yaw
                    self.detected_markers.append((marker_id, x, y, z, yaw))

                # Sort by distance (z, closest first)
                self.detected_markers.sort(key=lambda m: m[3])

        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")

    def odom_callback(self, msg):
        self.odom_linear_vel = msg.twist.twist.linear.x
        self.odom_angular_vel = msg.twist.twist.angular.z

    def cmd_callback(self, msg):
        self.cmd_linear_vel = msg.linear.x
        self.cmd_angular_vel = msg.angular.z

    def lidar_callback(self, msg):
        self.current_lidar = [round(r, 3) for r in msg.ranges[::10]]  # every 10th point

    # --- Save data at fixed rate ---
    def save_data(self):
        if self.current_image is None:
            return

        # Save image
        img_name = f"{datetime.now().strftime('%H%M%S_%f')}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        cv2.imwrite(img_path, self.current_image)

        # Prepare marker data (up to 3 markers)
        markers_out = []
        for i in range(3):
            if i < len(self.detected_markers):
                markers_out.extend(self.detected_markers[i])
            else:
                markers_out.extend([None, None, None, None, None])

        # Save row
        self.csv_writer.writerow([
            img_name,
            self.odom_linear_vel, self.odom_angular_vel,
            self.cmd_linear_vel, self.cmd_angular_vel,
            self.current_lidar,
            *markers_out
        ])
        self.get_logger().info(f"Saved {img_name}")

    def destroy_node(self):
        self.csv_file.close()
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
