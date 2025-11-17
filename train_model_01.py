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
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt 
from ncps.tf import CfC


# =============================== CONFIG =====================================
CSV_PATH   = "/home/newin/Projects/warehouse/dataset_clear/goal _A/data_Home2A_ep1__clean.csv"     # <-- set me
IMAGE_DIR  = "/home/newin/Projects/warehouse/dataset_clear/goal _A/images"    # <-- set me
# Output / logs
#SAVE_PATH = "/home/newin/Projects/warehouse/models/lnn_ltc_best.keras"
#LOG_DIR   = "/home/newin/Projects/warehouse/log_dir/lnn_ltc"

# ---- One place to name this run ----
RUN_FOLDER = "home2A_seq32_cfc64_tr_01"   # <— change this per run

# Root folders
ROOT_MODEL_DIR = "/home/newin/Projects/warehouse/models"
ROOT_LOG_DIR   = "/home/newin/Projects/warehouse/log_dir"

# ---- Create run folders & paths ----
MODEL_DIR = os.path.join(ROOT_MODEL_DIR, RUN_FOLDER)
LOG_DIR   = os.path.join(ROOT_LOG_DIR,   RUN_FOLDER)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SAVE_PATH = os.path.join(MODEL_DIR, f"{RUN_FOLDER}.keras")
WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{RUN_FOLDER}.weights.h5")


print(f"[OUT] Model dir : {MODEL_DIR}")
print(f"[OUT] Log dir   : {LOG_DIR}")
print(f"[OUT] Model file: {SAVE_PATH}")

# If your raw LiDAR was pre-clipped/scaled, keep this for reference
LIDAR_MAX_RANGE = 10.0   # set to your actual preprocessor range, or None
# Data/Model hyperparameters
SEQ_LEN        = 32
STRIDE         = 1
BATCH_SIZE     = 8
EPOCHS         = 50
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


# ---------------- Evaluation plots ----------------


def plot_train_val_mae(history, out_dir, fname="train_val_mae.png"):
    """Plot Train vs Val MAE per epoch."""
    mae = history.history.get("mae", [])
    vmae = history.history.get("val_mae", [])
    epochs = range(1, len(mae)+1)

    plt.figure()
    plt.plot(epochs, mae, label="train_mae")
    plt.plot(epochs, vmae, label="val_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Train vs Val MAE per Epoch")
    plt.legend()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[PLOT] {path}")

def plot_val_train_gap(history, out_dir, fname="val_train_mae_gap.png"):
    """Plot (val_mae - mae) per epoch as an overfitting signal."""
    mae = history.history.get("mae", [])
    vmae = history.history.get("val_mae", [])
    gap = [v - m for m, v in zip(mae, vmae)]
    epochs = range(1, len(gap)+1)

    plt.figure()
    plt.plot(epochs, gap)
    plt.xlabel("Epoch")
    plt.ylabel("Val MAE - Train MAE")
    plt.title("Val–Train MAE Gap per Epoch")
    path = os.path.join(out_dir, fname)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[PLOT] {path}")

def plot_per_timestep_mae(model, val_ds, seq_len, out_dir, fname="per_timestep_val_mae.png"):
    mt = compute_per_timestep_mae(model, val_ds, seq_len)
    steps = range(1, len(mt)+1)

    plt.figure()
    plt.plot(steps, mt)
    plt.xlabel("Timestep (1 … SEQ_LEN)")
    plt.ylabel("Val MAE")
    plt.title("Per-timestep Validation MAE")
    path = os.path.join(out_dir, fname)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[PLOT] {path}")

"""def compute_per_timestep_mae(model, val_ds, seq_len):
    #Return array shape (seq_len,) with mean MAE per timestep over the whole val set.
    import numpy as np
    
    sums = np.zeros(seq_len, dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.int64)



    for (img_seq, lidar_seq, state_seq), y_true in val_ds:
        y_pred = model.predict_on_batch([img_seq, lidar_seq, state_seq])  # (B,T,2)
        err = np.abs(y_pred - y_true).mean(axis=2)  # (B,T)
        sums += err.sum(axis=0)
        counts += err.shape[0]
    
    return sums / np.maximum(counts, 1)



def scatter_pred_vs_gt(model, val_ds, out_dir,
                       fname_linear="pred_vs_gt_linear.png",
                       fname_angular="pred_vs_gt_angular.png",
                       max_points=100000):
    
    #Save two scatter plots: linear and angular component.
    #Samples up to max_points points for speed/clarity.
    
    import numpy as np
    y_list = []
    p_list = []
    for (img_seq, lidar_seq, state_seq), y_true in val_ds:
        y_pred = model.predict_on_batch([img_seq, lidar_seq, state_seq])  # (B,T,2)
        y_list.append(y_true.numpy())
        p_list.append(y_pred)
    y = np.concatenate(y_list, axis=0).reshape(-1, 2)  # (N,2)
    p = np.concatenate(p_list, axis=0).reshape(-1, 2)  # (N,2)

    # Optionally subsample
    N = y.shape[0]
    if N > max_points:
        idx = np.random.default_rng(42).choice(N, size=max_points, replace=False)
        y = y[idx]; p = p[idx]

    # Linear velocity
    plt.figure()
    plt.scatter(y[:,0], p[:,0], s=2, alpha=0.4)
    lim = [min(y[:,0].min(), p[:,0].min()), max(y[:,0].max(), p[:,0].max())]
    plt.plot(lim, lim)  # y=x line
    plt.xlabel("GT linear")
    plt.ylabel("Pred linear")
    plt.title("Pred vs GT — Linear Velocity")
    path_lin = os.path.join(out_dir, fname_linear)
    plt.savefig(path_lin, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[PLOT] {path_lin}")

    # Angular velocity
    plt.figure()
    plt.scatter(y[:,1], p[:,1], s=2, alpha=0.4)
    lim = [min(y[:,1].min(), p[:,1].min()), max(y[:,1].max(), p[:,1].max())]
    plt.plot(lim, lim)
    plt.xlabel("GT angular")
    plt.ylabel("Pred angular")
    plt.title("Pred vs GT — Angular Velocity")
    path_ang = os.path.join(out_dir, fname_angular)
    plt.savefig(path_ang, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[PLOT] {path_ang}")

"""
def compute_per_timestep_mae(model, val_ds, seq_len):
    import numpy as np
    sums = np.zeros(seq_len, dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.int64)
    for (img_seq, lidar_seq, state_seq), y_true in val_ds:
        if hasattr(y_true, "numpy"):
            y_true = y_true.numpy()
        y_pred = model.predict_on_batch([img_seq, lidar_seq, state_seq])  # (B,T,2)
        err = np.abs(y_pred - y_true).mean(axis=2)  # (B,T)
        T = err.shape[1]
        sums[:T] += err.sum(axis=0)
        counts[:T] += err.shape[0]
    return sums / np.maximum(counts, 1)

def scatter_pred_vs_gt(model, val_ds, out_dir,
                       fname_linear="pred_vs_gt_linear.png",
                       fname_angular="pred_vs_gt_angular.png",
                       max_points=100000):
    import numpy as np, os, matplotlib.pyplot as plt
    y_list, p_list = [], []
    for (img_seq, lidar_seq, state_seq), y_true in val_ds:
        if hasattr(y_true, "numpy"):
            y_true = y_true.numpy()
        y_pred = model.predict_on_batch([img_seq, lidar_seq, state_seq])  # (B,T,2)
        y_list.append(y_true); p_list.append(y_pred)
    y = np.concatenate(y_list, axis=0).reshape(-1, 2)
    p = np.concatenate(p_list, axis=0).reshape(-1, 2)
    N = y.shape[0]
    if N > max_points:
        idx = np.random.default_rng(42).choice(N, size=max_points, replace=False)
        y, p = y[idx], p[idx]
    # linear
    plt.figure(); plt.scatter(y[:,0], p[:,0], s=2, alpha=0.4)
    lim = [min(y[:,0].min(), p[:,0].min()), max(y[:,0].max(), p[:,0].max())]
    plt.plot(lim, lim); plt.xlabel("GT linear"); plt.ylabel("Pred linear"); plt.title("Pred vs GT — Linear")
    plt.savefig(os.path.join(out_dir, fname_linear), bbox_inches="tight", dpi=150); plt.close()
    # angular
    plt.figure(); plt.scatter(y[:,1], p[:,1], s=2, alpha=0.4)
    lim = [min(y[:,1].min(), p[:,1].min()), max(y[:,1].max(), p[:,1].max())]
    plt.plot(lim, lim); plt.xlabel("GT angular"); plt.ylabel("Pred angular"); plt.title("Pred vs GT — Angular")
    plt.savefig(os.path.join(out_dir, fname_angular), bbox_inches="tight", dpi=150); plt.close()


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
    
    

    # ---- Combined run_info.json (includes scaler + config + paths) ----
    run_info = {
        "RUN_FOLDER": RUN_FOLDER,
        "paths": {
            "model_dir": MODEL_DIR,
            "log_dir": LOG_DIR,
            "save_path": SAVE_PATH,
            "csv": CSV_PATH,
            "image_dir": IMAGE_DIR,
        },
        "env": {
            "tensorflow": tf.__version__,
            "HAS_NCPS": bool(_HAS_NCPS),
            "RNN_CORE": "CfC" if _HAS_NCPS else "GRU",
        },
        "hyperparams": {
            "SEQ_LEN": SEQ_LEN, "STRIDE": STRIDE, "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS, "LEARNING_RATE": LEARNING_RATE, "VAL_SPLIT": VAL_SPLIT,
            "MIXED_PRECISION": MIXED_PRECISION, "JITTER": JITTER, "SEED": SEED,
            "IMG_SHAPE": list(IMG_SHAPE), "LIDAR_DIM": LIDAR_DIM,
            "STATE_DIM": STATE_DIM, "HIDDEN": HIDDEN,
        },
        # ---- scaler used during training (TRAIN-only μ/σ) ----
        "scaler": {
            "lidar_mean": stats["lidar_mean"].tolist(),
            "lidar_std":  stats["lidar_std"].tolist(),
            "state_mean": stats["state_mean"].tolist(),
            "state_std":  stats["state_std"].tolist(),
            "lidar_max_range": float(LIDAR_MAX_RANGE) if LIDAR_MAX_RANGE is not None else None
        }
    }

    run_info_path = os.path.join(MODEL_DIR, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"[SAVE] run_info.json -> {run_info_path}")


    
    # Callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=WEIGHTS_PATH,
        monitor="val_mae",
        mode="min",
        save_best_only=True,
        save_weights_only=True,   # <--- IMPORTANT
        verbose=1
    )
        
    es = tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=6, restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    tb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, write_graph=False)

    print(f"[RNN] Core selected: {'CfC' if _HAS_NCPS else 'GRU'}")

    print("[TRAIN] Starting…")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[ es, rlrop, tb],
        verbose=1
    )
    for (img_seq, lidar_seq, state_seq), y_true in val_ds.take(1):
        _ = model.predict([img_seq, lidar_seq, state_seq], verbose=0)
        break
    # Save final (best already saved)-------------------------------

    try:
        model.save(SAVE_PATH)
        print(f"[SAVE] Model saved to {SAVE_PATH}")
    except Exception as e:
        print("[SAVE] Failed:", e)
    
    model.save_weights(WEIGHTS_PATH)
    print(f"[SAVE] Weights saved to {WEIGHTS_PATH}")


    # ---- Save a human-readable run_config.txt in the model folder ----
    cfg_lines = [
        f"RUN_FOLDER      = {RUN_FOLDER}",
        f"CSV_PATH        = {CSV_PATH}",
        f"IMAGE_DIR       = {IMAGE_DIR}",
        "",
        "# Data/Model hyperparameters",
        f"SEQ_LEN         = {SEQ_LEN}",
        f"STRIDE          = {STRIDE}",
        f"BATCH_SIZE      = {BATCH_SIZE}",
        f"EPOCHS          = {EPOCHS}",
        f"LEARNING_RATE   = {LEARNING_RATE}",
        f"VAL_SPLIT       = {VAL_SPLIT}",
        f"MIXED_PRECISION = {MIXED_PRECISION}",
        f"JITTER          = {JITTER}",
        f"SEED            = {SEED}",
        "",
        "# Modalities",
        f"IMG_SHAPE       = {IMG_SHAPE}",
        f"LIDAR_DIM       = {LIDAR_DIM}",
        f"STATE_DIM       = {STATE_DIM}",
        f"HIDDEN          = {HIDDEN}",
        "",
        f"has NCP (CfC)   = {bool(_HAS_NCPS)}",
        f"RNN core        = {'CfC' if _HAS_NCPS else 'GRU'}",
        "",
        f"SAVE_PATH       = {SAVE_PATH}",
        f"LOG_DIR         = {LOG_DIR}",
        f"MODEL_DIR       = {MODEL_DIR}",
    ]
    with open(os.path.join(MODEL_DIR, "run_config.txt"), "w") as f:
        f.write("\n".join(cfg_lines))
    print(f"[SAVE] run_config.txt -> {os.path.join(MODEL_DIR, 'run_config.txt')}")

    # === Generate requested plots into MODEL_DIR ===
    plot_train_val_mae(history, MODEL_DIR)                      # Train vs Val MAE
    plot_val_train_gap(history, MODEL_DIR)                      # Val–Train MAE gap
    plot_per_timestep_mae(model, val_ds, SEQ_LEN, MODEL_DIR)    # Per-timestep Val MAE
    scatter_pred_vs_gt(model, val_ds, MODEL_DIR)                # Pred vs GT (linear & angular)


if __name__ == "__main__":
    main()
