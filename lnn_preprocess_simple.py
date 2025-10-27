#!/usr/bin/env python3
import os, re, ast, json
from typing import List, Tuple
import numpy as np
import pandas as pd

# ======================= CONFIG (EDIT THIS) =======================
INPUT_CSVS = [
    "/home/newin/Projects/warehouse/dataset_clear/goal _A/data_Home2A_ep1.csv",
    
]
OUT_DIR = "/home/newin/Projects/warehouse/dataset_clear/goal _A"
LIDAR_COL = "lidar_points"
LIDAR_MAX_RANGE = 10.0
KEEP_IMAGE_FILE = True
ROUND_DECIMALS = 4
# ================================================================

USE_COLUMNS_DROP = [
    "start_id", "goal_id",
    "marker0_id","marker0_x","marker0_y","marker0_z","marker0_yaw",
    "marker1_id","marker1_x","marker1_y","marker1_z","marker1_yaw",
    "marker2_id","marker2_x","marker2_y","marker2_z","marker2_yaw",
    "marker3_id","marker3_x","marker3_y","marker3_z","marker3_yaw",
]

CORE_FEATURE_COLS = [
    "x_r","y_r","yaw_r",
    "x_g","y_g","yaw_g",
    "dX","dY","sin_dYaw","cos_dYaw","dist_to_goal","dist_home_to_goal",
    "odom_linear_vel","odom_angular_vel",
]

TARGET_COLS = ["cmd_linear_vel","cmd_angular_vel"]
IMAGE_COL = "image_file"

def parse_lidar_cell(cell, max_range: float) -> np.ndarray:
    if isinstance(cell, (list, np.ndarray)):
        arr = np.array(cell, dtype=float)
    else:
        s = str(cell).strip()
        s = re.sub(r"\b(inf|Inf|INF|Infinity)\b", str(max_range), s)
        if not (s.startswith("[") and s.endswith("]")):
            s = "[" + s.strip("[]") + "]"
        try:
            py_list = ast.literal_eval(s)
        except Exception:
            toks = re.split(r"[,\s]+", s.strip("[]"))
            py_list = []
            for t in toks:
                if t == "":
                    continue
                if re.fullmatch(r"nan|NaN|NAN", t):
                    py_list.append(max_range)
                else:
                    try:
                        py_list.append(float(t))
                    except Exception:
                        py_list.append(max_range)
        arr = np.array(py_list, dtype=float)
    arr = np.nan_to_num(arr, nan=max_range, posinf=max_range, neginf=0.0)
    arr = np.clip(arr, 0.0, max_range)
    return arr

def expand_and_normalize(
    df: pd.DataFrame,
    lidar_col: str = "lidar_points",
    lidar_max_range: float = 10.0,
    keep_image_col: bool = True
):
    df_local = df.copy()
    drop_cols = [c for c in USE_COLUMNS_DROP if c in df_local.columns]
    if drop_cols:
        df_local = df_local.drop(columns=drop_cols)

    if lidar_col not in df_local.columns:
        raise ValueError(f"Expected lidar column '{lidar_col}' not found in CSV.")
    lidar_arrays = [parse_lidar_cell(v, max_range=lidar_max_range) for v in df_local[lidar_col]]
    lengths = pd.Series([len(a) for a in lidar_arrays])
    modal_len = int(lengths.mode().iloc[0])
    fixed_lidar = np.zeros((len(lidar_arrays), modal_len), dtype=float)
    for i, a in enumerate(lidar_arrays):
        if len(a) >= modal_len:
            fixed_lidar[i] = a[:modal_len]
        else:
            pad = np.full(modal_len - len(a), lidar_max_range, dtype=float)
            fixed_lidar[i] = np.concatenate([a, pad], axis=0)
    lidar_norm01 = fixed_lidar / float(lidar_max_range)
    lidar_cols = [f"lidar_{i}" for i in range(modal_len)]
    lidar_df = pd.DataFrame(lidar_norm01, columns=lidar_cols, index=df_local.index)
    df_local = pd.concat([df_local.drop(columns=[lidar_col]), lidar_df], axis=1)

    numeric_core = [c for c in CORE_FEATURE_COLS if c in df_local.columns]
    feature_cols = numeric_core + lidar_cols

    #  Keep episode_id if available
    if "episode_id" in df_local.columns:
        feature_cols = ["episode_id"] + feature_cols


    target_cols_present = [c for c in TARGET_COLS if c in df_local.columns]

    if keep_image_col and ("image_file" in df_local.columns):
        cleaned_df = df_local[[IMAGE_COL] + feature_cols + target_cols_present]
    else:
        cleaned_df = df_local[feature_cols + target_cols_present]

    X_raw = cleaned_df.drop(columns=[IMAGE_COL], errors="ignore").copy()
    y_cols = [c for c in target_cols_present]
    f_cols = [c for c in X_raw.columns if c not in y_cols]

    X_feat = X_raw[f_cols].astype(float).values
    mean = X_feat.mean(axis=0)
    std = X_feat.std(axis=0)
    std[std == 0.0] = 1.0
    X_norm = (X_feat - mean) / std

    norm_df = cleaned_df.copy()
    norm_df.loc[:, f_cols] = X_norm

    scaler = {
        "feature_order": f_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "note": "Z-score normalization for features only; targets untouched. LiDAR pre-scaled to [0,1] by max_range."
    }
    return cleaned_df, norm_df, scaler

def round_numeric(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(decimals)
    return out

def process_file(path: str, lidar_col: str, lidar_max_range: float, keep_image_col: bool, out_dir: str, decimals: int):
    df = pd.read_csv(path)
    clean_df, norm_df, scaler = expand_and_normalize(df, lidar_col=lidar_col, lidar_max_range=lidar_max_range, keep_image_col=keep_image_col)

    clean_df = round_numeric(clean_df, decimals)
    norm_df  = round_numeric(norm_df,  decimals)

    base = os.path.splitext(os.path.basename(path))[0]
    out_clean = os.path.join(out_dir, f"{base}__clean.csv")
    out_norm  = os.path.join(out_dir, f"{base}__norm.csv")
    out_scal  = os.path.join(out_dir, f"{base}__scaler.json")

    clean_df.to_csv(out_clean, index=False)
    norm_df.to_csv(out_norm, index=False)
    with open(out_scal, "w") as f:
        json.dump(scaler, f, indent=2)

    print(f"[OK] {path}\n -> {out_clean}\n -> {out_norm}\n -> {out_scal}")
    return out_clean, out_norm, out_scal

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for src in INPUT_CSVS:
        if not os.path.isfile(src):
            print(f"[WARN] Skipping missing file: {src}")
            continue
        try:
            process_file(src, LIDAR_COL, LIDAR_MAX_RANGE, KEEP_IMAGE_FILE, OUT_DIR, ROUND_DECIMALS)
        except Exception as e:
            print(f"[ERROR] {src}: {e}")

if __name__ == "__main__":
    main()