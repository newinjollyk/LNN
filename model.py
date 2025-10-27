# pip install ncps tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from ncps.tf import LTCCell   # <- LNN (Liquid) cell

# ---------- Image preprocessing (RGB -> gray 128x128) ----------
def preprocess_rgb_to_gray128(x):
    x = tf.image.rgb_to_grayscale(tf.cast(x, tf.float32))          # (H,W,1)
    x = tf.image.resize(x, (128, 128), method="bilinear") / 255.0   # (128,128,1)
    return x

# ---------- Encoders ----------
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

def build_goal_state_mlp(state_dim=12, out_dim=32):
    state_in = layers.Input(shape=(state_dim,), name="state_input")
    x = layers.Dense(32, activation="relu")(state_in)
    x = layers.Dense(out_dim, activation="relu", name="state_feature")(x)           # -> 32
    return models.Model(state_in, x, name="GoalState_MLP")

# ---------- Full sequence model with LNN core (LTCCell) ----------
def build_sequence_lnn_ltc(
    IMG_SHAPE=(128, 128, 1),
    LIDAR_DIM=67,
    STATE_DIM=12,
    LIQUID_HIDDEN=64
):
    # Inputs are sequences: (batch, time, ...)
    img_seq   = layers.Input(shape=(None,)+IMG_SHAPE, name="image_seq")
    lidar_seq = layers.Input(shape=(None, LIDAR_DIM), name="lidar_seq")
    state_seq = layers.Input(shape=(None, STATE_DIM), name="state_seq")

    # Per-timestep encoders
    cnn_enc   = build_cnn_encoder(IMG_SHAPE)
    lidar_enc = build_lidar_mlp(LIDAR_DIM)
    state_enc = build_goal_state_mlp(STATE_DIM, out_dim=32)

    img_feat_seq   = layers.TimeDistributed(cnn_enc,   name="TD_CNN")(img_seq)      # (B,T,64)
    lidar_feat_seq = layers.TimeDistributed(lidar_enc, name="TD_LiDAR")(lidar_seq)  # (B,T,64)
    state_feat_seq = layers.TimeDistributed(state_enc, name="TD_State")(state_seq)  # (B,T,32)

    # Fuse → 160 features per step, then a small pre-fusion MLP
    fused = layers.Concatenate(axis=-1, name="fuse")([img_feat_seq, lidar_feat_seq, state_feat_seq])  # (B,T,160)
    fused = layers.TimeDistributed(layers.Dense(128, activation="relu"), name="pre_liquid")(fused)     # (B,T,128)

    # Liquid Neural Network core (LTC)
    liquid = layers.RNN(LTCCell(LIQUID_HIDDEN), return_sequences=True, name="LTC_Layer")(fused)        # (B,T,64)

    # Head → velocities per step
    head = layers.TimeDistributed(layers.Dense(32, activation="relu"), name="post_liquid")(liquid)     # (B,T,32)
    out  = layers.TimeDistributed(layers.Dense(2, activation="linear"), name="vel_output")(head)       # (B,T,2)

    return models.Model(inputs=[img_seq, lidar_seq, state_seq], outputs=out, name="CNN_LiDAR_State_LTC")

# ---------- Build & compile ----------
model = build_sequence_lnn_ltc(
    IMG_SHAPE=(128,128,1),
    LIDAR_DIM=67,
    STATE_DIM=12,     # (xr, yr, yawr, xg, yg, yawg, dX, dY, sin_dYaw, cos_dYaw, dist_to_goal, <one more if any>)
    LIQUID_HIDDEN=64
)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
model.summary()
