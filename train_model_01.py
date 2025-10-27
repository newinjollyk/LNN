#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train CNN+LiDAR+GoalState -> LNN (LTC) with *no CLI arguments*.
Edit the constants in the CONFIG section below and run:
    python train_model_01.py
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# =============================== CONFIG =====================================
CSV_PATH   = "/home/newin/Projects/warehouse/dataset_clear/goal _A/data_Home2A_ep1__clean.csv"     # <-- set me
IMAGE_DIR  = "/home/newin/Projects/warehouse/dataset_clear/goal _A/images"    # <-- set me
# Output / logs
SAVE_PATH = "/home/newin/Projects/warehouse/models/lnn_ltc_best.keras"
LOG_DIR   = "/home/newin/Projects/warehouse/log_dir/lnn_ltc"

# Data/Model hyperparameters
SEQ_LEN        = 32
STRIDE         = 1
BATCH_SIZE     = 8
EPOCHS         = 20
LEARNING_RATE  = 1e-3
VAL_SPLIT      = 0.1
MIXED_PRECISION= True      # set False if you hit dtype issues on CPU
JITTER         = True      # light brightness/contrast jitter on images
SEED           = 42


# Modalities
IMG_SHAPE = (128, 128, 1)
LIDAR_DIM = 67
STATE_DIM = 11             # you asked for 11-dim goal/state
HIDDEN = 64


# ===========================================================================


# Try to import CfC (Liquid) from ncps.tf. If unavailable, we'll fall back to GRU.
_HAS_NCPS = False
try:
    from ncps.tf import CfC  # Keras RNN layer
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False

AUTOTUNE = tf.data.AUTOTUNE

# ---------------- GPU & precision setup ----------------
def setup_gpu(mixed_precision: bool):
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        if gpus:
            print(f"[GPU] Visible GPUs: {len(gpus)}")
    except Exception as e:
        print("[GPU] note:", e)
    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy("mixed_float16")
            print("[MP] Mixed precision enabled.")
        except Exception as e:
            print("[MP] Could not enable mixed precision:", e)

# ---------------- Episode-safe index builder ----------------
def episode_index_matrix(ep_ids: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Return (num_seq, seq_len) indices; never cross episodes; pad the tail by repeating last row."""
    N, out = len(ep_ids), []
    i = 0
    while i < N:
        j = i + 1
        while j < N and ep_ids[j] == ep_ids[i]:
            j += 1
        L = j - i
        # rolling windows
        start, made_any = i, False
        while start + seq_len <= j:
            out.append(np.arange(start, start + seq_len, dtype=np.int32))
            made_any = True
            start += stride
        # tail / short episode padding
        if L < seq_len:
            base = np.arange(i, j, dtype=np.int32)
            pad = np.full(seq_len - L, j - 1, dtype=np.int32)
            out.append(np.concatenate([base, pad]))
        else:
            if not made_any or (j - (start - stride + seq_len)) > 0:
                s = max(j - seq_len, i)
                seq = np.arange(s, s + seq_len, dtype=np.int32)
                seq = np.minimum(seq, j - 1)
                if not made_any or s != (start - stride):
                    out.append(seq)
        i = j
    return np.stack(out, axis=0) if out else np.empty((0, seq_len), dtype=np.int32)

# ---------------- Image preprocessing ----------------
def decode_to_gray_128(img_bytes: tf.Tensor) -> tf.Tensor:
    img = tf.image.decode_image(img_bytes, channels=0, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    c = tf.shape(img)[-1]
    img = tf.cond(
        tf.equal(c, 3),
        lambda: tf.image.rgb_to_grayscale(img),
        lambda: tf.cond(tf.equal(c, 4),
                        lambda: tf.image.rgb_to_grayscale(img[..., :3]),
                        lambda: img)
    )
    img = tf.image.resize(img, IMG_SHAPE[:2], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.ensure_shape(img, IMG_SHAPE)
    return img

def load_and_preprocess_image(path: tf.Tensor, jitter: bool) -> tf.Tensor:
    img = decode_to_gray_128(tf.io.read_file(path))
    if jitter:
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img

# ---------------- Columns ----------------
LIDAR_COLS = [f"lidar_{i}" for i in range(LIDAR_DIM)]
# Pick 11 state features: prior 12 minus 'dist_home_to_goal'
STATE_COLS = [
    "x_r","y_r","yaw_r",
    "x_g","y_g","yaw_g",
    "dX","dY","sin_dYaw","cos_dYaw",
    "dist_to_goal"
]
TARGET_COLS = ["cmd_linear_vel", "cmd_angular_vel"]

# ---------------- Dataset builder ----------------
def build_datasets(csv_path: str,
                   image_dir: str,
                   seq_len: int,
                   stride: int,
                   batch_size: int,
                   val_split: float,
                   seed: int,
                   jitter: bool):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    def col(name): 
        k = name.lower()
        if k not in cols:
            raise KeyError(f"Missing column '{name}'. Found: {list(df.columns)}")
        return cols[k]

    img_col = col("image_file")
    ep_col  = col("episode_id")

    # paths
    def resolve_path(fn: str) -> str:
        return os.path.normpath(os.path.join(image_dir, os.path.basename(fn.strip().replace("\\","/"))))
    image_paths = [resolve_path(p) for p in df[img_col].astype(str)]
    ep_ids = df[ep_col].to_numpy()

    # numeric arrays
    lidar_np = df[[col(c) for c in LIDAR_COLS]].astype(np.float32).to_numpy()
    state_np = df[[col(c) for c in STATE_COLS]].astype(np.float32).to_numpy()
    y_np     = df[[col(c) for c in TARGET_COLS]].astype(np.float32).to_numpy()

    # sequences (episode-safe with tail padding)
    seq_index = episode_index_matrix(ep_ids, seq_len, stride)
    if seq_index.shape[0] == 0:
        raise RuntimeError("No sequences formed. Check SEQ_LEN/STRIDE vs episode length.")

    # split on sequences
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(seq_index))
    seq_index = seq_index[perm]
    val_n = int(len(seq_index) * val_split)
    val_idx = seq_index[:val_n]
    train_idx = seq_index[val_n:]

    # standardize LiDAR & state using TRAIN rows only
    train_rows = np.unique(train_idx.reshape(-1))
    eps = 1e-6
    lidar_mean, lidar_std = lidar_np[train_rows].mean(0), lidar_np[train_rows].std(0) + eps
    state_mean, state_std = state_np[train_rows].mean(0), state_np[train_rows].std(0) + eps

    lidar_norm = (lidar_np - lidar_mean) / lidar_std
    state_norm = (state_np - state_mean) / state_std

    # to tf constants
    paths_t  = tf.constant(image_paths)
    lidar_t  = tf.constant(lidar_norm, dtype=tf.float32)
    state_t  = tf.constant(state_norm, dtype=tf.float32)
    target_t = tf.constant(y_np, dtype=tf.float32)

    AUTOTUNE = tf.data.AUTOTUNE

    def make_mapper(is_training: bool):
        def _map(index_vec: tf.Tensor):
            seq_paths = tf.gather(paths_t, index_vec)
            img_seq = tf.map_fn(
                lambda p: load_and_preprocess_image(p, jitter=is_training and jitter),
                seq_paths,
                fn_output_signature=tf.TensorSpec(shape=IMG_SHAPE, dtype=tf.float32),
                parallel_iterations=16,
            )
            lidar_seq = tf.gather(lidar_t,  index_vec)   # (T,67)
            state_seq = tf.gather(state_t,  index_vec)   # (T,11)
            y_seq     = tf.gather(target_t, index_vec)   # (T,2)
            return (img_seq, lidar_seq, state_seq), y_seq
        return _map

    def as_ds(index_mat: np.ndarray):
        ds = tf.data.Dataset.from_tensor_slices(index_mat)
        ds = ds.shuffle(buffer_size=max(1, len(index_mat)), seed=seed, reshuffle_each_iteration=True)
        return ds

    train_ds = (as_ds(train_idx)
                .map(make_mapper(True), num_parallel_calls=AUTOTUNE)
                .batch(batch_size, drop_remainder=True)
                .prefetch(AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices(val_idx)
              .map(make_mapper(False), num_parallel_calls=AUTOTUNE)
              .batch(batch_size, drop_remainder=False)
              .prefetch(AUTOTUNE))

    stats = {
        "num_sequences_total": int(len(seq_index)),
        "num_sequences_train": int(len(train_idx)),
        "num_sequences_val":   int(len(val_idx)),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "lidar_mean": lidar_mean, "lidar_std": lidar_std,
        "state_mean": state_mean, "state_std": state_std,
    }
    return train_ds, val_ds, stats

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

# -------------------------- Train script --------------------------

def main():
    # Repro
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Safety checks
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.isdir(IMAGE_DIR):
        raise NotADirectoryError(f"Image dir not found: {IMAGE_DIR}")

    setup_gpu(MIXED_PRECISION)

    # Data
    train_ds, val_ds, stats = build_datasets(
        
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED,
        jitter=JITTER,
    )
    print("[DATA]", {k: (v if isinstance(v, (int,float)) else type(v).__name__) for k,v in stats.items()})

    # Model
    
    model = build_sequence_lnn_cfc(IMG_SHAPE=IMG_SHAPE, LIDAR_DIM=LIDAR_DIM, STATE_DIM=STATE_DIM, HIDDEN=HIDDEN)

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    model.summary()

    # Callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=SAVE_PATH, monitor="val_mae", mode="min",
        save_best_only=True, save_weights_only=False, verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=6, restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    tb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=False)

    print("[TRAIN] Starting…")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ckpt, es, rlrop, tb],
        verbose=1
    )

    # Save final (best already saved)
    try:
        model.save(SAVE_PATH)
        print(f"[SAVE] Model saved to {SAVE_PATH}")
    except Exception as e:
        print("[SAVE] Failed:", e)

if __name__ == "__main__":
    main()
