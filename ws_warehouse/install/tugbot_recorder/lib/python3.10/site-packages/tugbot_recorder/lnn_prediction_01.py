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

import tensorflow as tf
from tensorflow.keras import layers, models

# ===================== USER SETTINGS (edit before each run) =====================
START_ID   = 'Home'   # not used in inference, kept for consistency
GOAL_ID    = 'A'      # <-- choose: 'Home' / 'A' / 'B' / 'C'
SEQ_LEN    = 32
FPS        = 10.0     # target inference rate (Hz) once buffers are full

# Model run selection (mirrors your training script structure)
RUN_FOLDER     = "home2A_seq32_cfc64_tr_01"    # <-- change this per run
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


# Try to import CfC (Liquid) from ncps.tf. If unavailable, we'll fall back to GRU.
_HAS_NCPS = False
try:
    from ncps.tf import CfC  # Keras RNN layer
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False

AUTOTUNE = tf.data.AUTOTUNE

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

# ---------------- Your model blocks ----------------

def build_cnn_encoder(input_shape=(128, 128, 1)):
    img_in = layers.Input(shape=input_shape, name="image_input")
    x = layers.Conv2D(16, 5, strides=2, padding="same", activation="relu")(img_in)  # 64x64x16
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)       # 32x32x32
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)       # 16x16x64
    x = layers.Flatten()(x)                                                         # 16*16*64=16384
    x = layers.Dense(64, activation="relu", name="cnn_feature")(x)                  # -> 64
    return models.Model(img_in, x, name="CNN_Encoder")

def build_lidar_mlp(lidar_dim=67):
    lidar_in = layers.Input(shape=(lidar_dim,), name="lidar_input")
    x = layers.Dense(64, activation="relu")(lidar_in)
    x = layers.Dense(64, activation="relu", name="lidar_feature")(x)                # -> 64
    return models.Model(lidar_in, x, name="LiDAR_MLP")

def build_goal_state_mlp(state_dim=11, out_dim=32):
    state_in = layers.Input(shape=(state_dim,), name="state_input")
    x = layers.Dense(32, activation="relu")(state_in)
    x = layers.Dense(out_dim, activation="relu", name="state_feature")(x)           # -> 32
    return models.Model(state_in, x, name="GoalState_MLP")

def build_sequence_lnn_cfc(
    IMG_SHAPE=(128,128,1), LIDAR_DIM=67, STATE_DIM=11, HIDDEN=64
):
    img_seq   = tf.keras.layers.Input(shape=(None,)+IMG_SHAPE, name="image_seq")
    lidar_seq = tf.keras.layers.Input(shape=(None, LIDAR_DIM),  name="lidar_seq")
    state_seq = tf.keras.layers.Input(shape=(None, STATE_DIM),  name="state_seq")

    # TimeDistributed encoders (same idea as the official code)
    cnn_enc   = build_cnn_encoder(IMG_SHAPE)         # -> 64 dims per frame
    lidar_enc = build_lidar_mlp(LIDAR_DIM)           # -> 64
    state_enc = build_goal_state_mlp(STATE_DIM, 32)  # -> 32

    img_feat   = tf.keras.layers.TimeDistributed(cnn_enc,   name="TD_CNN")(img_seq)      # (B,T,64)
    lidar_feat = tf.keras.layers.TimeDistributed(lidar_enc, name="TD_LiDAR")(lidar_seq)  # (B,T,64)
    state_feat = tf.keras.layers.TimeDistributed(state_enc, name="TD_State")(state_seq)  # (B,T,32)

    fused = tf.keras.layers.Concatenate(axis=-1, name="fuse")([img_feat, lidar_feat, state_feat])  # (B,T,160)
    fused = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"),
                                            name="pre_rnn")(fused)  # (B,T,128)

    # ---- CfC core (official style: RNN layer with units=int) ----
    
    if _HAS_NCPS:
        rnn = CfC(units=HIDDEN, return_sequences=True, name="cfc")
    else:
        rnn = tf.keras.layers.GRU(units=HIDDEN, return_sequences=True, name="gru")
    x = rnn(fused)  # (B,T,HIDDEN)

    # Head -> (B,T,2)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="relu"), name="post_rnn")(x)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2), name="vel_output")(x)

    return tf.keras.Model(inputs=[img_seq, lidar_seq, state_seq], outputs=out, name="CNN_LiDAR_State_CfC")


class InferenceNode(Node):
    def __init__(self):
        super().__init__('lnn_inference')

        # ---- Build paths from RUN_FOLDER (matches your training layout) ----
        self.MODEL_DIR = os.path.join(ROOT_MODEL_DIR, RUN_FOLDER)
        self.LOG_DIR   = os.path.join(ROOT_LOG_DIR,   RUN_FOLDER)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR,   exist_ok=True)

        self.model_path    = os.path.join(self.MODEL_DIR, f"{RUN_FOLDER}.keras")
        self.run_info_path = os.path.join(self.MODEL_DIR, "run_info.json")

        self.get_logger().info(f"[MODEL] {self.model_path}")
        self.get_logger().info(f"[INFO ] {self.run_info_path}")

        # ---- Load scaler + (optional) lidar max range from run_info.json ----
 
        with open(self.run_info_path, "r") as f:
            run_info = json.load(f)
        scaler = run_info["scaler"]
        hparams = run_info["hyperparams"]

        self.lidar_mean = np.array(scaler["lidar_mean"], dtype=np.float32)
        self.lidar_std  = np.array(scaler["lidar_std"],  dtype=np.float32)
        self.state_mean = np.array(scaler["state_mean"], dtype=np.float32)
        self.state_std  = np.array(scaler["state_std"],  dtype=np.float32)
        self.lidar_max_range = float(scaler.get("lidar_max_range", 10.0))

        # Rebuild the **same** architecture used during training
        
        IMG_SHAPE = tuple(hparams["IMG_SHAPE"])
        LIDAR_DIM = hparams["LIDAR_DIM"]
        STATE_DIM = hparams["STATE_DIM"]
        HIDDEN    = hparams["HIDDEN"]

        self.lidar_dim = LIDAR_DIM

        self.model = build_sequence_lnn_cfc(
            IMG_SHAPE=IMG_SHAPE,
            LIDAR_DIM=LIDAR_DIM,
            STATE_DIM=STATE_DIM,
            HIDDEN=HIDDEN,
        )

        # Load the trained weights
        WEIGHTS_PATH = os.path.join(self.MODEL_DIR, f"{RUN_FOLDER}.weights.h5")
        self.get_logger().info(f"[MODEL] Loading weights from: {WEIGHTS_PATH}")
        self.model.load_weights(WEIGHTS_PATH)

        # ---- Buffers (rolling) ----
        self.seq_len = SEQ_LEN
        self.buf_img   = deque(maxlen=self.seq_len)  # each (128,128) float32
        self.buf_lidar = deque(maxlen=self.seq_len)  # each (67,)    float32
        self.buf_state = deque(maxlen=self.seq_len)  # each (11,)    float32

        # ---- ROS setup: same topics as recorder.py ----
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

        # Predicted velocity publisher (Ignition/Gazebo bridge-friendly)
        """
        self.twist_pub = self.create_publisher(
            Twist,
            '/model/tugbot/Prediction_vel',  # you can visualize/bridge like /model/tugbot/cmd_vel
            10
        )
        """
        self.twist_pub = self.create_publisher(
            Twist,
            '/model/tugbot/cmd_vel',
            10
        )

        # Optional: also publish raw predictions for logging
        self.vec_pub = self.create_publisher(Float32MultiArray, '/lnn/prediction', 10)

        # Run inference at 10 Hz when buffers are ready
        self.timer = self.create_timer(1.0 / FPS, self.inference_step)

        # ---- Goal selection ----
        assert GOAL_ID in GOALS, "GOAL_ID must be one of: " + ", ".join(GOALS.keys())
        g = GOALS[GOAL_ID]
        self.goal_pose_world = (float(g['x']), float(g['y']), math.radians(float(g['yaw_deg'])))
        self.get_logger().info(f"[GOAL] Using hardcoded goal '{GOAL_ID}': {self.goal_pose_world}")

        # Robot pose cache (world)
        self.robot_pose_world = (0.0, 0.0, 0.0)

    # ---------- Callbacks ----------
    def image_callback(self, msg: Image):
        """Convert to grayscale 128x128 and scale to [0,1]."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
            img_f  = (resized.astype(np.float32) / 255.0)
            self.buf_img.append(img_f)
        except Exception as e:
            self.get_logger().error(f"image_callback error: {e}")

    def lidar_callback(self, msg: LaserScan):
        """Resample LiDAR to lidar_dim, clip to max range, scale to [0,1], then z-score."""
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            n_raw  = ranges.shape[0]
            target = self.lidar_dim  # e.g. 67, from run_info.json

            # 1) Downsample / upsample to exactly `target` beams:
            if n_raw == target:
                lidar = ranges
            else:
                # Choose `target` evenly spaced indices across the scan
                idx = np.linspace(0, n_raw - 1, target, dtype=int)
                lidar = ranges[idx]

            # 2) Replace inf/nan with max range
            lidar = np.nan_to_num(
                lidar,
                nan=self.lidar_max_range,
                posinf=self.lidar_max_range,
                neginf=0.0,
            )

            # 3) Clip and normalize [0,1]
            lidar = np.clip(lidar, 0.0, float(self.lidar_max_range))
            lidar01 = lidar / float(self.lidar_max_range)

            # 4) z-score with same stats as training
            lidar_z = zscore(lidar01, self.lidar_mean, self.lidar_std).astype(np.float32)

            self.buf_lidar.append(lidar_z)

        except Exception as e:
            self.get_logger().error(f"lidar_callback error: {e}")


    def odom_callback(self, msg: Odometry):
        """Cache robot pose and push z-scored 11-D state vector."""
        try:
            x_r = float(msg.pose.pose.position.x)
            y_r = float(msg.pose.pose.position.y)
            yaw_r = quat_to_yaw(msg.pose.pose.orientation)
            self.robot_pose_world = (x_r, y_r, yaw_r)

            state_z = self._build_state_z(self.robot_pose_world, self.goal_pose_world)
            self.buf_state.append(state_z)
        except Exception as e:
            self.get_logger().error(f"odom_callback error: {e}")

    # ---------- Preprocessing for state ----------
    def _build_state_z(self, robot_pose_world, goal_pose_world):
        """
        Build the 11-D state vector exactly as in training/recorder:
        [x_r, y_r, yaw_r, x_g, y_g, yaw_g, dX, dY, sin_dYaw, cos_dYaw, dist_to_goal]
        with dist_to_goal computed in WORLD coords via dist2d((x_r,y_r),(x_g,y_g)).
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

    # ---------- Inference ----------
    def inference_step(self):
        """Run the model when all buffers have SEQ_LEN frames; publish as Twist."""
        if len(self.buf_img) < self.seq_len or len(self.buf_lidar) < self.seq_len or len(self.buf_state) < self.seq_len:
            return

        try:
            # Stack and add batch dimension: (1, T, ...)
            img_seq   = np.stack(self.buf_img,   axis=0)[None, ...]          # (1, T, 128, 128)
            lidar_seq = np.stack(self.buf_lidar, axis=0)[None, ...]          # (1, T, 67)
            state_seq = np.stack(self.buf_state, axis=0)[None, ...]          # (1, T, 11)

            # If your model expects channels for image, add channel dim:
            if img_seq.ndim == 4:  # (1, T, H, W)
                img_seq = img_seq[..., None]  # (1, T, H, W, 1)

            # Predict: assume output shape (1, 2) -> [v_lin, v_ang]
            pred = self.model.predict([img_seq, lidar_seq, state_seq], verbose=0)
            v_lin = float(pred[0, -1, 0])
            v_ang = float(pred[0, -1, 1])

            # Publish Twist (Ignition/Gazebo bridge-compatible)
            tw = Twist()
            tw.linear.x  = v_lin
            tw.angular.z = v_ang
            self.twist_pub.publish(tw)

            # Optional: also publish raw vector
            msg = Float32MultiArray()
            msg.data = [v_lin, v_ang]
            self.vec_pub.publish(msg)
            """
            Twist:
            ros2 topic echo /model/tugbot/Prediction_vel geometry_msgs/msg/Twist

            # Raw vector:
            ros2 topic echo /lnn/prediction std_msgs/msg/Float32MultiArray
            """
        except Exception as e:
            self.get_logger().error(f"inference_step error: {e}")

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
