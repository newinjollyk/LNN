#!/usr/bin/env python3
import os
import json
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from collections import deque

from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

# ===================== USER SETTINGS (edit before each run) =====================
START_ID   = 'Home'   # kept for consistency
GOAL_ID    = 'A'      # <-- 'Home' / 'A' / 'B' / 'C'
SEQ_LEN    = 32       # sequence length required by the model
FPS        = 10.0     # <-- single knob: sampling + inference rate (Hz)

# Model run selection (matches your training structure)
RUN_FOLDER     = "home2A_seq32_cfc64"    # <-- change per run
ROOT_MODEL_DIR = "/home/newin/Projects/warehouse/models"
ROOT_LOG_DIR   = "/home/newin/Projects/warehouse/log_dir"
# ==============================================================================

# World/map frame coordinates from Ignition Gazebo
GOALS = {
    'Home': {'x':  13.900000, 'y': -10.600000, 'yaw_deg': -0.000003},
    'A':    {'x': -13.638600, 'y':   2.687560, 'yaw_deg': -3.115870},
    'B':    {'x': -13.630200, 'y':  10.011400, 'yaw_deg': -3.113550},
    'C':    {'x': -13.636200, 'y':  17.522800, 'yaw_deg': -3.114620},
}

# ---------- Helpers ----------
def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def relative_goal(robot_pose, goal_pose):
    """ robot_pose=(x_r,y_r,yaw_r), goal_pose=(x_g,y_g,yaw_g) in WORLD.
        Returns dX,dY in ROBOT frame and sin/cos(dYaw). """
    x_r, y_r, th_r = robot_pose
    x_g, y_g, th_g = goal_pose
    dx = x_g - x_r
    dy = y_g - y_r
    c, s = math.cos(th_r), math.sin(th_r)
    dX =  c*dx + s*dy
    dY = -s*dx + c*dy
    dYaw = wrap_to_pi(th_g - th_r)
    return dX, dY, math.sin(dYaw), math.cos(dYaw)

def dist2d(p1, p2) -> float:
    """ Euclidean distance in WORLD frame (same as recorder.py). """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def zscore(x, mean, std):
    return (x - mean) / std


class InferenceNode(Node):
    def __init__(self):
        super().__init__('lnn_inference')

        # ---- Build paths from RUN_FOLDER (your training layout) ----
        self.MODEL_DIR = os.path.join(ROOT_MODEL_DIR, RUN_FOLDER)
        self.LOG_DIR   = os.path.join(ROOT_LOG_DIR,   RUN_FOLDER)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR,   exist_ok=True)

        self.model_path    = os.path.join(self.MODEL_DIR, f"{RUN_FOLDER}.keras")
        self.run_info_path = os.path.join(self.MODEL_DIR, "run_info.json")

        self.get_logger().info(f"[MODEL] {self.model_path}")
        self.get_logger().info(f"[INFO ] {self.run_info_path}")

        # ---- Load scaler (+ lidar_max_range) from run_info.json ----
        with open(self.run_info_path, "r") as f:
            run_info = json.load(f)
        scaler = run_info["scaler"]
        self.lidar_mean = np.array(scaler["lidar_mean"], dtype=np.float32)  # len=67
        self.lidar_std  = np.array(scaler["lidar_std"],  dtype=np.float32)
        self.state_mean = np.array(scaler["state_mean"], dtype=np.float32)  # len=11
        self.state_std  = np.array(scaler["state_std"],  dtype=np.float32)
        self.lidar_max_range = scaler.get("lidar_max_range", 10.0) or 10.0  # fallback

        # ---- Load model ----
        from tensorflow.keras.models import load_model
        self.model = load_model(self.model_path)

        # ---- Buffers (rolling) ----
        self.seq_len = int(SEQ_LEN)
        self.buf_img   = deque(maxlen=self.seq_len)  # (128,128) float32
        self.buf_lidar = deque(maxlen=self.seq_len)  # (67,)    float32
        self.buf_state = deque(maxlen=self.seq_len)  # (11,)    float32

        # ---- ROS setup: use EXACT SAME TOPICS as recorder.py ----
        self.bridge = CvBridge()

        self.latest_img   = None
        self.latest_lidar = None
        self.latest_odom  = None

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

        # Predicted velocity publisher (Ignition/Gazebo bridge-friendly)
        self.twist_pub = self.create_publisher(
            Twist,
            '/model/tugbot/Prediction_vel',  # consumer topic in Ignition
            10
        )
        # Optional: also publish raw vector for logging
        self.vec_pub = self.create_publisher(Float32MultiArray, '/lnn/prediction', 10)

        # ---- Goal selection ----
        assert GOAL_ID in GOALS, "GOAL_ID must be one of: " + ", ".join(GOALS.keys())
        g = GOALS[GOAL_ID]
        self.goal_pose_world = (float(g['x']), float(g['y']), math.radians(float(g['yaw_deg'])))
        self.get_logger().info(f"[GOAL] Using hardcoded goal '{GOAL_ID}': {self.goal_pose_world}")

        # ---- Single timer drives sampling + inference at FPS ----
        self.fps = float(FPS) if FPS and FPS > 0 else 10.0
        self.sample_timer = self.create_timer(1.0 / self.fps, self.sample_and_infer_step)

    # ---------- Callbacks: only cache latest msgs; no preprocess here ----------
    def image_callback(self, msg: Image):
        self.latest_img = msg

    def lidar_callback(self, msg: LaserScan):
        self.latest_lidar = msg

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    # ---------- Periodic sampler (runs at FPS) + inference ----------
    def sample_and_infer_step(self):
        # Need all three streams to sample a frame
        if self.latest_img is None or self.latest_lidar is None or self.latest_odom is None:
            return

        # --- SAMPLE at FPS: preprocess now ---
        # image
        try:
            img = self._preprocess_image_from_msg(self.latest_img)
            self.buf_img.append(img)
        except Exception as e:
            self.get_logger().warn(f"image preprocess skipped: {e}")
            return

        # lidar
        try:
            lidar_z = self._preprocess_lidar_from_msg(self.latest_lidar)
            self.buf_lidar.append(lidar_z)
        except Exception as e:
            self.get_logger().warn(f"lidar preprocess skipped: {e}")
            return

        # state
        try:
            x_r = float(self.latest_odom.pose.pose.position.x)
            y_r = float(self.latest_odom.pose.pose.position.y)
            yaw_r = quat_to_yaw(self.latest_odom.pose.pose.orientation)
            robot_pose_world = (x_r, y_r, yaw_r)
            state_z = self._build_state_z(robot_pose_world, self.goal_pose_world)
            self.buf_state.append(state_z)
        except Exception as e:
            self.get_logger().warn(f"state preprocess skipped: {e}")
            return

        # --- INFER at FPS when buffers are ready ---
        if len(self.buf_img) < self.seq_len or len(self.buf_lidar) < self.seq_len or len(self.buf_state) < self.seq_len:
            return

        try:
            # Shape to model inputs with batch dim
            img_seq   = np.stack(self.buf_img,   axis=0)[None, ...]  # (1, T, 128, 128)
            lidar_seq = np.stack(self.buf_lidar, axis=0)[None, ...]  # (1, T, 67)
            state_seq = np.stack(self.buf_state, axis=0)[None, ...]  # (1, T, 11)

            # If model expects channel dim for images:
            if img_seq.ndim == 4:  # (1, T, H, W)
                img_seq = img_seq[..., 1:] if img_seq.shape[-1] == 1 else img_seq[..., None]  # ensure (1,T,H,W,1)
                if img_seq.shape[-1] != 1:
                    img_seq = img_seq[..., None]

            # Predict: assume output shape (1, 2) -> [v_lin, v_ang]
            pred = self.model.predict([img_seq, lidar_seq, state_seq], verbose=0)
            v_lin, v_ang = float(pred[0][0]), float(pred[0][1])

            # Publish Twist (Ignition/Gazebo bridge-compatible)
            tw = Twist()
            tw.linear.x  = v_lin
            tw.angular.z = v_ang
            self.twist_pub.publish(tw)

            # Optional vector log
            arr = Float32MultiArray(); arr.data = [v_lin, v_ang]
            self.vec_pub.publish(arr)

        except Exception as e:
            self.get_logger().error(f"inference error: {e}")

    # ---------- Preprocess helpers ----------
    def _preprocess_image_from_msg(self, img_msg):
        """Grayscale 128x128, scale to [0,1]."""
        cv_img  = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        gray    = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)

    def _preprocess_lidar_from_msg(self, lidar_msg):
        """Subsample to 67, replace NaN/inf, clip to max, scale [0,1], z-score."""
        lidar = np.array(lidar_msg.ranges, dtype=np.float32)[::10]  # 67 beams
        lidar = np.nan_to_num(lidar, nan=self.lidar_max_range, posinf=self.lidar_max_range, neginf=0.0)
        lidar = np.clip(lidar, 0.0, float(self.lidar_max_range))
        lidar01 = lidar / float(self.lidar_max_range)
        lidar_z = zscore(lidar01, self.lidar_mean, self.lidar_std).astype(np.float32)
        return lidar_z

    def _build_state_z(self, robot_pose_world, goal_pose_world):
        """
        [x_r, y_r, yaw_r, x_g, y_g, yaw_g, dX, dY, sin_dYaw, cos_dYaw, dist_to_goal]
        dist_to_goal computed in WORLD coords via dist2d((x_r,y_r),(x_g,y_g)).
        """
        x_r, y_r, yaw_r = robot_pose_world
        x_g, y_g, yaw_g = goal_pose_world

        dX, dY, sin_dYaw, cos_dYaw = relative_goal(robot_pose_world, goal_pose_world)
        dist_to_goal = dist2d((x_r, y_r), (x_g, y_g))

        state = np.array([
            x_r, y_r, yaw_r,
            x_g, y_g, yaw_g,
            dX, dY, sin_dYaw, cos_dYaw,
            dist_to_goal
        ], dtype=np.float32)

        return zscore(state, self.state_mean, self.state_std).astype(np.float32)


# ---------- Main ----------
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
