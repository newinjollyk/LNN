
import os
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# Try to import CfC (Liquid) from ncps.tf. If unavailable, we'll fall back to GRU.
_HAS_NCPS = False
try:
    from ncps.tf import CfC  # Keras RNN layer
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False

AUTOTUNE = tf.data.AUTOTUNE

def resolve_path(base_dir: str, p: str) -> str:
    """
    Resolve image path from CSV (Windows or POSIX), mapping it under base_dir.
    Strategy:
      - Normalize slashes and trim whitespace
      - If 'IMG/' appears, take the suffix from 'IMG/' (case-insensitive) and join with base_dir
      - Else if relative -> join with base_dir
      - Else (absolute path without 'IMG/') -> keep as-is and try basename index later
    """
    p0 = p.strip().replace("\\", "/")
    low = p0.lower()
    idx = low.rfind("img/")
    if idx >= 0:
        suffix = p0[idx:]  # e.g., IMG/center_...jpg
        return os.path.normpath(os.path.join(base_dir, suffix))
    if not os.path.isabs(p0):
        return os.path.normpath(os.path.join(base_dir, p0))
    return p0

def crop_and_resize(image, out_w=200, out_h=66):
    """Crop sky and hood (NVIDIA-style) and resize."""
    # image is float32 [0,1], shape [H,W,3]
    shape = tf.shape(image)
    h = shape[0]
    top = tf.cast(tf.math.round(tf.cast(h, tf.float32) * 0.35), tf.int32)
    bottom = h - tf.cast(tf.math.round(tf.cast(h, tf.float32) * 0.10), tf.int32)
    image = image[top:bottom, :, :]
    image = tf.image.resize(image, [out_h, out_w], method=tf.image.ResizeMethod.BILINEAR)
    return image

def parse_and_preprocess(path, jitter=True):
    """Read image file -> float32 [0,1] -> crop/resize -> (H,W,3)."""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = crop_and_resize(img)
    if jitter:
        # Simple per-frame brightness/contrast jitter
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img

def build_dataset(
    data_dir: str,
    csv_path: str,
    seq_len: int = 20,
    stride: int = 1,
    batch_size: int = 32,
    camera_mode: str = "random_three",
    lr_steer_corr: float = 0.2,
    flip: bool = True,
    jitter: bool = True,
    val_split: float = 0.1,
    shuffle_buffer: int = 2048,
    seed: int = 42,
):
    # Load CSV (case-insensitive column resolution)
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    def need(name):
        if name in cols:
            return cols[name]
        raise ValueError(f"CSV missing required column '{name}'. Found: {list(df.columns)}")

    col_center = cols.get("centercam", cols.get("center", need("centercam")))
    col_left   = cols.get("leftcam",   cols.get("left", need("leftcam")))
    col_right  = cols.get("rightcam",  cols.get("right", need("rightcam")))
    col_steer  = cols.get("steering_angle", cols.get("steering", need("steering_angle")))
    col_reverse = cols.get("reverse", None)

    # Optional speed (not required for training steering)
    col_speed = cols.get("speed", None)

    # Build basename index for fallback
    file_index = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for root, dirs, files in os.walk(data_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                file_index[fn.lower()] = os.path.join(root, fn)

    def map_with_index(p: str) -> str:
        rp = resolve_path(data_dir, p)
        if os.path.exists(rp):
            return rp
        bn = os.path.basename(p).lower()
        if bn in file_index and os.path.exists(file_index[bn]):
            return file_index[bn]
        return rp  # may not exist; tf.data will error if truly missing

    center_paths = [map_with_index(p) for p in df[col_center].astype(str).tolist()]
    left_paths   = [map_with_index(p) for p in df[col_left].astype(str).tolist()]
    right_paths  = [map_with_index(p) for p in df[col_right].astype(str).tolist()]
    steering     = df[col_steer].astype(np.float32).to_numpy()
    speed        = df[col_speed].astype(np.float32).to_numpy() if (col_speed is not None) else None
    reverse      = df[col_reverse].astype(int).to_numpy() if (col_reverse is not None) else None

    # Drop reverse rows if present
    if reverse is not None:
        keep = reverse == 0
        center_paths = list(np.array(center_paths)[keep])
        left_paths   = list(np.array(left_paths)[keep])
        right_paths  = list(np.array(right_paths)[keep])
        steering     = steering[keep]
        if speed is not None:
            speed = speed[keep]

    # Build sequence starts
    N = len(steering)
    starts = np.arange(0, N - seq_len + 1, stride, dtype=np.int32)

    # Constants to tensors for tf.data
    center_t = tf.constant(center_paths)
    left_t   = tf.constant(left_paths)
    right_t  = tf.constant(right_paths)
    steer_t  = tf.constant(steering)
    if speed is not None:
        speed_t = tf.constant(speed)
    else:
        speed_t = None

    # Dataset of sequence start indices
    ds = tf.data.Dataset.from_tensor_slices(starts)

    # Shuffle and split
    ds = ds.shuffle(buffer_size=len(starts), seed=seed, reshuffle_each_iteration=True)
    val_size = int(len(starts) * val_split)
    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)

    def make_mapper(is_training: bool):
        def mapper(start_idx):
            start_idx = tf.cast(start_idx, tf.int32)
            idx = tf.range(start_idx, start_idx + seq_len, dtype=tf.int32)

            # Choose camera for the whole sequence
            if camera_mode == "center" or (not is_training):
                cam_sel = tf.constant(0, dtype=tf.int32)  # 0=center
            else:
                r = tf.random.uniform([], 0.0, 1.0)
                cam_sel = tf.where(r < 0.6, 0, tf.where(r < 0.8, 1, 2))  # 0=center,1=left,2=right

            # Gather paths and steering
            paths_center = tf.gather(center_t, idx)
            paths_left   = tf.gather(left_t, idx)
            paths_right  = tf.gather(right_t, idx)

            steer_seq = tf.gather(steer_t, idx)

            # Apply steering correction for left/right cameras
            corr = tf.cast(tf.cond(tf.equal(cam_sel, 1),
                                   lambda: lr_steer_corr,
                                   lambda: tf.cond(tf.equal(cam_sel, 2), lambda: -lr_steer_corr, lambda: 0.0)),
                           tf.float32)
            steer_seq = steer_seq + corr

            # Select camera paths
            def sel_center(): return paths_center
            def sel_left():   return paths_left
            def sel_right():  return paths_right

            paths_sel = tf.switch_case(cam_sel, branch_fns={0: sel_center, 1: sel_left, 2: sel_right})

            # Read & preprocess all images
            imgs = tf.map_fn(
                lambda p: parse_and_preprocess(p, jitter=is_training and jitter),
                paths_sel,
                fn_output_signature=tf.TensorSpec(shape=(66, 200, 3), dtype=tf.float32),
                parallel_iterations=16
            )

            # Optional horizontal flip (sequence-wise) with steering inversion
            if flip and is_training:
                do_flip = tf.less(tf.random.uniform([]), 0.5)
                imgs = tf.cond(do_flip, lambda: tf.image.flip_left_right(imgs), lambda: imgs)
                steer_seq = tf.cond(do_flip, lambda: -steer_seq, lambda: steer_seq)

            # Targets shape: (T,1)
            steer_seq = tf.expand_dims(steer_seq, axis=-1)

            return imgs, steer_seq
        return mapper

    train = (train_ds
             .map(make_mapper(is_training=True), num_parallel_calls=AUTOTUNE)
             .shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
             .batch(batch_size, drop_remainder=True)
             .prefetch(AUTOTUNE))

    val = (val_ds
           .map(make_mapper(is_training=False), num_parallel_calls=AUTOTUNE)
           .batch(batch_size, drop_remainder=False)
           .prefetch(AUTOTUNE))

    return train, val, len(starts)

def build_model(seq_len=20, hidden_dim=128, use_cfc=True):
    inputs = tf.keras.Input(shape=(seq_len, 66, 200, 3), name="images")

    # TimeDistributed CNN encoder (NVIDIA-like)
    def cnn_block():
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 5, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(36, 5, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(48, 5, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu"),
            tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(128, activation=None),
        ], name="cnn_encoder")

    x = tf.keras.layers.TimeDistributed(cnn_block(), name="td_encoder")(inputs)

    # Liquid core (CfC) or GRU
    if use_cfc:
        rnn = CfC(units=hidden_dim, return_sequences=True, name="cfc")
    else:
        rnn = tf.keras.layers.GRU(units=hidden_dim, return_sequences=True, name="gru")
    x = rnn(x)

    # Steering head
    steer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu"), name="steer_fc")(x)
    steer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1), name="steering")(steer)

    model = tf.keras.Model(inputs=inputs, outputs=steer, name="LNN_CfC_Udacity_TF")
    return model

def main():
    """
    Argument Guide:
    -------------------------------------------------------------------------
    | Argument        | Default / Example     | Meaning                                         | Recommended Start Value | When to Change                          |
    |-----------------|-----------------------|-------------------------------------------------|-------------------------|-----------------------------------------|
    | --seq_len       | 20                    | Frames per training sample (temporal context).  | 20 (≈ 2 s at 10 Hz)      | 10–15 for speed, 25–30 for more context |
    | --stride        | 1                     | Step to slide window for next sequence.         | 1                        | 2–5 to speed up, less overlap           |
    | --camera_mode   | random_three          | Which camera(s) to use.                         | random_three             | center for no augmentation              |
    | --lr_steer_corr | 0.2                   | Steering correction for left/right images.      | 0.2                      | 0.25–0.3 if understeering; 0.15 if over |
    | --no_jitter     | (flag)                | Disable brightness/contrast augmentation.       | Leave unset              | Set for raw images/debugging            |
    | --no_flip       | (flag)                | Disable horizontal flip augmentation.           | Leave unset              | Set if flip causes issues               |
    | --val_split     | 0.1                   | Fraction of data for validation.                | 0.1 (10%)                | 0.2 if lots of data                     |
    | --save_path     | lnn_udacity_tf.keras  | File to save best model.                         | lnn_udacity_tf.keras     | Change for separate runs                |
    -------------------------------------------------------------------------
    """
    parser = argparse.ArgumentParser(description="Train TensorFlow Liquid (CfC) model on Udacity Behavioral Cloning")
    parser.add_argument("--data_dir", type=str, default= "/home/newin/Projects/warehouse/lnn/self_driving_car_dataset_jungle")
    parser.add_argument("--csv", type=str, default= "/home/newin/Projects/warehouse/lnn/self_driving_car_dataset_jungle/driving_log.csv")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--camera_mode", type=str, default="random_three", choices=["center","random_three"])
    parser.add_argument("--lr_steer_corr", type=float, default=0.2)
    parser.add_argument("--no_jitter", action="store_true")
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default="/home/newin/Projects/warehouse/lnn/lnn_udacity_tf.keras")
    args = parser.parse_args()

    print("TensorFlow version:", tf.__version__)
    print("ncps available (CfC):", _HAS_NCPS)

    train_ds, val_ds, nseq = build_dataset(
        data_dir=args.data_dir,
        csv_path=args.csv,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        camera_mode=args.camera_mode,
        lr_steer_corr=args.lr_steer_corr,
        flip=(not args.no_flip),
        jitter=(not args.no_jitter),
        val_split=args.val_split,
    )

    model = build_model(seq_len=args.seq_len, hidden_dim=128, use_cfc=_HAS_NCPS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.save_path,
        monitor="val_mae",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=5, restore_best_weights=True)

    steps_per_epoch = None
    validation_steps = None

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt, es],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )

    # Save final model as well (best is saved via checkpoint)
    try:
        model.save(args.save_path)
        print(f"Saved model to {args.save_path}")
    except Exception as e:
        print("Save failed:", e)

if __name__ == "__main__":
    main()
