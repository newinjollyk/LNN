import os
import json
import numpy as np
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from collections import deque

# Z-Score Normalization (for LiDAR and State)
def zscore(data, mean, std):
    return (data - mean) / std

# Helper to convert quaternion to yaw angle
def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

# Compute relative goal coordinates (robot to goal transformation)
def relative_goal(robot_pose, goal_pose):
    x_r, y_r, th_r = robot_pose
    x_g, y_g, th_g = goal_pose
    dx = x_g - x_r
    dy = y_g - y_r
    c, s = math.cos(th_r), math.sin(th_r)
    dX = c * dx + s * dy  # relative x in robot frame
    dY = -s * dx + c * dy  # relative y in robot frame
    dYaw = wrap_to_pi(th_g - th_r)  # wrap the yaw difference
    return dX, dY, math.sin(dYaw), math.cos(dYaw)

# Helper function to wrap angles between -pi and +pi
def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

# Load the normalization stats (mean and std for lidar and state)
def load_scaler(scaler_path):
    with open(scaler_path, "r") as f:
        scaler = json.load(f)
    return scaler
# Preprocess the robot goal-conditioned state
def dist2d(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Hardcoded goal positions (this replaces the dynamic goal pose subscription)
GOALS = {
    'Home': {'x': 13.900000, 'y': -10.600000, 'yaw_deg': -0.000003},
    'A': {'x': -13.638600, 'y': 2.687560, 'yaw_deg': -3.115870},
    'B': {'x': -13.630200, 'y': 10.011400, 'yaw_deg': -3.113550},
    'C': {'x': -13.636200, 'y': 17.522800, 'yaw_deg': -3.114620},
}

class InferenceNode(Node):
    def __init__(self):
        super().__init__('lnn_inference')
        
        # Hardcoded topic names
        self.camera_topic = '/world/world_demo/model/tugbot/link/camera_front/sensor/color/image'
        self.lidar_topic = '/world/world_demo/model/tugbot/link/scan_front/sensor/scan_front/scan'
        self.odom_topic = '/model/tugbot/odometry'
        
        # Sequence buffer size
        self.seq_len = 32
        self.stride = 1
        
        # Dynamically get the RUN_FOLDER (change per run)
        RUN_FOLDER = "home2A_seq32_cfc64"  # Change this for each run

        # Root folders
        ROOT_MODEL_DIR = "/home/newin/Projects/warehouse/models"
        ROOT_LOG_DIR = "/home/newin/Projects/warehouse/log_dir"

        # Create run paths
        self.MODEL_DIR = os.path.join(ROOT_MODEL_DIR, RUN_FOLDER)
        self.LOG_DIR = os.path.join(ROOT_LOG_DIR, RUN_FOLDER)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

        self.SAVE_PATH = os.path.join(self.MODEL_DIR, f"{RUN_FOLDER}.keras")
        self.get_logger().info(f"[OUT] Model dir : {self.MODEL_DIR}")
        self.get_logger().info(f"[OUT] Log dir   : {self.LOG_DIR}")
        self.get_logger().info(f"[OUT] Model file: {self.SAVE_PATH}")

        # Paths to the scaler file and model
        self.scaler_path = os.path.join(self.MODEL_DIR, "run_info.json")
        self.model_path = self.SAVE_PATH  # This will point to the model file

        # Load scaler data (mean/std for LiDAR and state)
        scaler = load_scaler(self.scaler_path)
        self.lidar_mean = np.array(scaler["scaler"]["lidar_mean"])
        self.lidar_std = np.array(scaler["scaler"]["lidar_std"])
        self.state_mean = np.array(scaler["scaler"]["state_mean"])
        self.state_std = np.array(scaler["scaler"]["state_std"])

        # Buffers to store data sequences
        self.buf_img = deque(maxlen=self.seq_len)
        self.buf_lid = deque(maxlen=self.seq_len)
        self.buf_state = deque(maxlen=self.seq_len)

        # Create subscriber for topics
        self.bridge = CvBridge()
        
        self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback, 10
        )
        self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback, 10
        )
        self.create_subscription(
            LaserScan,
            self.lidar_topic,
            self.lidar_callback, 10
        )
        
        # Publisher for predictions
        self.pub = self.create_publisher(Float32MultiArray, '/lnn/prediction', 10)

    # Image callback function (preprocess)
    def image_callback(self, msg):
        img = self.preprocess_image(msg)
        self.buf_img.append(img)

    # LiDAR callback function (preprocess)
    def lidar_callback(self, msg):
        lidar = self.preprocess_lidar(msg)
        self.buf_lid.append(lidar)

    # Odometry callback function (compute robot pose)
    def odom_callback(self, msg):
        robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, quat_to_yaw(msg.pose.pose.orientation))
        
        # Hardcoded goal pose (we'll use the goal from GOALS)
        goal_pose = (GOALS['A']['x'], GOALS['A']['y'], math.radians(GOALS['A']['yaw_deg']))  # Example: Goal A
        
        # Preprocess the goal state
        state = self.preprocess_goal(robot_pose, goal_pose)
        self.buf_state.append(state)

    # Preprocess the image (grayscale, resize, normalize)
    def preprocess_image(self, img_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        return resized.astype(np.float32) / 255.0

    # Preprocess LiDAR data (resample, clip, normalize, z-score)
    def preprocess_lidar(self, lidar_msg):
        lidar = np.array(lidar_msg.ranges)[::10]  # Subsample to 67 points
        lidar = np.clip(lidar, 0, 10.0)  # Clip to max range (10 meters)
        lidar = lidar / 10.0  # Scale to [0, 1]
        return zscore(lidar, self.lidar_mean, self.lidar_std)

    

    # Updated preprocess_goal function
    def preprocess_goal(self, robot_pose, goal_pose):
        # Calculate relative goal position (robot frame) - we will omit this in the updated code
        dX, dY, sin_dYaw, cos_dYaw = relative_goal(robot_pose, goal_pose)

        # Global positions (x_r, y_r) for the robot and (x_g, y_g) for the goal
        x_r, y_r, yaw_r = robot_pose
        x_g, y_g, yaw_g = goal_pose

        # Calculate distance to goal using dist2d() function (global coordinates)
        dist_to_goal = dist2d((x_r, y_r), (x_g, y_g))  # Euclidean distance between robot and goal

        # Now, create the goal state (we exclude dist_home_to_goal)
        state = np.array([x_r, y_r, yaw_r,  # robot position and yaw
                        x_g, y_g, yaw_g,  # goal position and yaw
                        dX, dY, sin_dYaw, cos_dYaw,
                        dist_to_goal])  # distance to goal calculated using dist2d()

        # Return z-scored state
        return zscore(state, self.state_mean, self.state_std)

    # Inference step (run model when all buffers are full)
    def inference_step(self):
        if len(self.buf_img) == self.seq_len and len(self.buf_lid) == self.seq_len and len(self.buf_state) == self.seq_len:
            img_seq = np.stack(self.buf_img, axis=0)
            lidar_seq = np.stack(self.buf_lid, axis=0)
            state_seq = np.stack(self.buf_state, axis=0)
            
            # Run inference (model predictions)
            prediction = self.model.predict([img_seq, lidar_seq, state_seq])
            
            # Publish predicted velocities
            prediction_msg = Float32MultiArray()
            prediction_msg.data = prediction.flatten().tolist()
            self.pub.publish(prediction_msg)

# Run the node
def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()