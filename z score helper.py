import json, numpy as np

def load_scaler_from_run(run_info_path):
    with open(run_info_path, "r") as f:
        info = json.load(f)
    s = info["scaler"]
    lidar_mean = np.array(s["lidar_mean"], dtype=np.float32)
    lidar_std  = np.array(s["lidar_std"],  dtype=np.float32)
    state_mean = np.array(s["state_mean"], dtype=np.float32)
    state_std  = np.array(s["state_std"],  dtype=np.float32)
    lidar_max  = s.get("lidar_max_range", None)
    return (lidar_mean, lidar_std, state_mean, state_std, lidar_max)

def apply_scaler(lidar_np, state_np, scaler):
    lidar_mean, lidar_std, state_mean, state_std, lidar_max = scaler
    lidar = lidar_np.astype(np.float32)
    state = state_np.astype(np.float32)
    # if you also do [0,1] clipping at serve time, do it before z-score:
    if lidar_max is not None:
        lidar = np.clip(lidar, 0.0, float(lidar_max)) / float(lidar_max)
    lidar_z = (lidar - lidar_mean) / lidar_std
    state_z = (state - state_mean) / state_std
    return lidar_z, state_z

# usage:
# scaler = load_scaler_from_run("/.../models/<RUN_FOLDER>/run_info.json")
# lidar_z, state_z = apply_scaler(lidar_np, state_np, scaler)
