import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from keras.models import load_model  # <— no tensorflow import
from keras.models import load_model
from ncps.tf import CfC

# -------------------------------
# Preprocess exactly like training
# -------------------------------
def crop_and_resize_pil(img: Image.Image, out_w=200, out_h=66) -> Image.Image:
    w, h = img.size
    top = int(h * 0.35)             # remove sky
    bottom = h - int(h * 0.10)      # remove hood
    img = img.crop((0, top, w, bottom))
    return img.resize((out_w, out_h), Image.BILINEAR)

def load_and_preprocess_image(img_path, target=(66, 200)):
    img = Image.open(img_path).convert("RGB")
    img = crop_and_resize_pil(img, out_w=target[1], out_h=target[0])  # (H,W)
    arr = np.asarray(img).astype(np.float32) / 255.0                  # [0,1]
    return arr  # (66,200,3)

def main():
    parser = argparse.ArgumentParser(description="Predict steering + show CSV labels for selected Udacity images")
    parser.add_argument("--model", type=str, required=True, help="Path to trained Keras model (.keras)")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root (folder that contains IMG/)")
    parser.add_argument("--csv", type=str, required=True, help="Path to driving_log.csv")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length used during training")
    parser.add_argument("--images", nargs="+", required=True, help="List of absolute image paths to predict")
    parser.add_argument("--out_csv", type=str, default=None, help="Output CSV for predictions (default: predictions.csv next to model)")
    args = parser.parse_args()

    # Load CSV; add headers if missing
    df = pd.read_csv(args.csv)
    if not {"centercam","center","leftcam","left","rightcam","right"}.intersection({c.lower() for c in df.columns}):
        # Typical Udacity order
        df.columns = ["centercam","leftcam","rightcam","steering_angle","throttle","reverse","speed"][:df.shape[1]]
    cols = {c.lower(): c for c in df.columns}
    col_center = cols.get("centercam", cols.get("center"))
    col_left   = cols.get("leftcam", cols.get("left"))
    col_right  = cols.get("rightcam", cols.get("right"))
    col_steer  = cols.get("steering_angle", cols.get("steering"))
    col_thr    = cols.get("throttle", None)
    col_rev    = cols.get("reverse", cols.get("brake", None))
    col_spd    = cols.get("speed", None)

    # Normalize CSV paths to actual files under data_dir
    def resolve_path(p: str) -> str:
        p0 = str(p).strip().replace("\\", "/")
        low = p0.lower()
        idx = low.rfind("img/")
        suffix = p0[idx:] if idx >= 0 else p0  # prefer IMG/... suffix
        base = args.data_dir
        if base.lower().rstrip("/").endswith("img") and suffix.lower().startswith("img/"):
            base = os.path.dirname(base)  # avoid IMG/IMG
        return os.path.normpath(os.path.join(base, suffix))

    center_paths = [resolve_path(p) for p in df[col_center].astype(str)] if col_center else []
    left_paths   = [resolve_path(p) for p in df[col_left].astype(str)]   if col_left   else []
    right_paths  = [resolve_path(p) for p in df[col_right].astype(str)]  if col_right  else []

    # Build quick lookups by basename
    def to_map(paths):
        m = {}
        for i, p in enumerate(paths):
            bn = os.path.basename(p).lower()
            m[bn] = i
        return m
    map_center = to_map(center_paths) if center_paths else {}
    map_left   = to_map(left_paths)   if left_paths   else {}
    map_right  = to_map(right_paths)  if right_paths  else {}

    print(f"Loading model from {args.model} …")
    model = load_model(
    "/home/newin/Projects/warehouse/lnn/lnn_udacity_tf.keras",
    compile=False,
    custom_objects={
        "CfC": CfC,          # usual registered name
        "ncps>CfC": CfC,     # some models serialize with this scoped name
    })
    

    records = []
    for full_path in args.images:
        bn = os.path.basename(full_path).lower()
        idx = None
        cam = None
        if bn in map_center:
            idx = map_center[bn]; cam = "center"
        elif bn in map_left:
            idx = map_left[bn]; cam = "left"
        elif bn in map_right:
            idx = map_right[bn]; cam = "right"
        else:
            print(f"[WARN] {bn} not found in CSV; skipping.")
            continue

        # Build sequence ending at idx
        T = args.seq_len
        start = max(0, idx - (T - 1))
        indices = list(range(start, idx + 1))
        if len(indices) < T:
            indices = [indices[0]] * (T - len(indices)) + indices

        # Prefer center frames; fallback to the chosen camera; then any that exists
        seq_paths = []
        for j in indices:
            candidates = []
            if center_paths:
                candidates.append(center_paths[j])
            if left_paths:
                candidates.append(left_paths[j])
            if right_paths:
                candidates.append(right_paths[j])
            path = next((p for p in candidates if p and os.path.exists(p)), None)
            if path is None:
                raise FileNotFoundError(f"No image found for CSV row {j}")
            seq_paths.append(path)

        # Load sequence tensor: (1, T, 66, 200, 3)
        seq = np.stack([load_and_preprocess_image(p) for p in seq_paths], axis=0)[None, ...]

        # Predict (model returns (B,T,1) in our training setup; take last)
        pred = model.predict(seq, verbose=0)
        # Handle both shapes (B, T, 1) or (B, 1) depending on saved model
        if pred.ndim == 3:
            pred_steer = float(pred[0, -1, 0])
        else:
            pred_steer = float(pred[0, 0])

        row = df.iloc[idx]
        gt_steer = float(row[col_steer]) if col_steer else np.nan
        thr = float(row[col_thr]) if col_thr else np.nan
        rev = int(row[col_rev]) if col_rev else 0
        spd = float(row[col_spd]) if col_spd else np.nan

        print(f"{bn} -> pred_steer={pred_steer:.4f} | gt={gt_steer:.4f} | thr={thr} | rev={rev} | spd={spd}")
        records.append({
            "image": os.path.basename(full_path),
            "row_index": idx,
            "matched_camera": cam,
            "predicted_steering": pred_steer,
            "gt_steering_angle": gt_steer,
            "throttle": thr,
            "reverse": rev,
            "speed": spd
        })

    if records:
        out_csv = args.out_csv or os.path.join(os.path.dirname(args.model) or ".", "predictions.csv")
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print("Saved predictions to:", out_csv)
    else:
        print("No predictions produced (no images matched the CSV).")

if __name__ == "__main__":
    main()
