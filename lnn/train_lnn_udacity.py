
import os
import math
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Try to load CfC (Liquid) from ncps. If unavailable, we'll fall back to GRU.
_HAS_NCPS = False
try:
    from ncps.torch import CfCCell  # type: ignore
    _HAS_NCPS = True
except Exception:
    _HAS_NCPS = False

# -----------------------------
# Utilities
# -----------------------------

def resolve_path(base_dir: str, p: str) -> str:
    """Resolve image path from CSV (which may contain absolute or relative paths)."""
    p = p.strip()
    if os.path.isabs(p):
        return p
    # Some Udacity logs have paths like 'IMG/center_2016_12_01_13_30_48_287.jpg'
    return os.path.normpath(os.path.join(base_dir, p))

def crop_and_resize(img: Image.Image, out_w: int = 200, out_h: int = 66) -> Image.Image:
    """Crop sky and hood, then resize (NVIDIA-style)."""
    w, h = img.size
    top = int(h * 0.35)     # remove sky
    bottom = h - int(h * 0.10)  # remove car hood
    img = img.crop((0, top, w, bottom))
    img = img.resize((out_w, out_h), Image.BILINEAR)
    return img

def to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image (RGB) to normalized float tensor (C,H,W) in [0,1]."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    # H,W,C -> C,H,W
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)

def color_jitter(img: Image.Image, brightness=0.2, contrast=0.2) -> Image.Image:
    """Simple brightness/contrast jitter."""
    # Brightness
    if brightness > 0:
        factor_b = 1.0 + random.uniform(-brightness, brightness)
        img = Image.fromarray(np.clip(np.asarray(img).astype(np.float32) * factor_b, 0, 255).astype(np.uint8))
    # Contrast
    if contrast > 0:
        mean = np.mean(np.asarray(img).astype(np.float32), axis=(0,1), keepdims=True)
        factor_c = 1.0 + random.uniform(-contrast, contrast)
        arr = (np.asarray(img).astype(np.float32) - mean) * factor_c + mean
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img

def horizontal_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)

# -----------------------------
# Dataset
# -----------------------------

class UdacitySeqDataset(Dataset):
    """
    Builds fixed-length sequences from the Udacity Behavioral Cloning dataset.

    Expected columns (case-insensitive):
      - centercam, leftcam, rightcam (file paths)
      - steering_angle (float)
      - throttle (float) [unused]
      - reverse (0/1)   [optional, unused]
      - speed (float)   [optional if you want to train speed too]

    We create sequences with length T using a sliding window over rows.
    Camera mode:
      - 'center'       : always center camera
      - 'random_three' : pick one of {center,left,right} for the *entire* sequence, apply +/- correction to steering
    """
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        seq_len: int = 20,
        stride: int = 1,
        camera_mode: str = "center",
        lr_steer_correction: float = 0.2,
        use_speed_label: bool = False,
        jitter: bool = True,
        flip_prob: float = 0.5,
        drop_reverse: bool = True,
        target_fps: float = 10.0,   # used to create constant dt if timestamps are absent
        center_bias_drop_p: float = 0.0 # probability to drop a sequence with near-zero steering (handled in __len__/__getitem__)
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.stride = stride
        self.camera_mode = camera_mode
        self.lr_corr = lr_steer_correction
        self.use_speed_label = use_speed_label
        self.jitter = jitter
        self.flip_prob = flip_prob
        self.center_bias_drop_p = center_bias_drop_p

        # Load CSV (robust to slight naming differences)
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}

        def need(name):
            # map a desired lowercase name to existing column
            if name in cols:
                return cols[name]
            raise ValueError(f"CSV missing required column '{name}'. Found columns: {list(df.columns)}")

        # Column names (fallbacks if different names are used)
        col_center = cols.get("centercam", cols.get("center", need("centercam")))
        col_left   = cols.get("leftcam",   cols.get("left", need("leftcam")))
        col_right  = cols.get("rightcam",  cols.get("right", need("rightcam")))
        col_steer  = cols.get("steering_angle", cols.get("steering", need("steering_angle")))
        col_speed  = cols.get("speed", None)
        col_reverse = cols.get("reverse", None)

        # Clean paths and extract arrays
        self.center_paths = [resolve_path(data_dir, p) for p in df[col_center].astype(str).tolist()]
        self.left_paths   = [resolve_path(data_dir, p) for p in df[col_left].astype(str).tolist()]
        self.right_paths  = [resolve_path(data_dir, p) for p in df[col_right].astype(str).tolist()]
        self.steering     = df[col_steer].astype(np.float32).to_numpy()
        self.speed        = df[col_speed].astype(np.float32).to_numpy() if (self.use_speed_label and (col_speed is not None)) else None
        self.reverse      = df[col_reverse].astype(int).to_numpy() if (col_reverse is not None) else None

        # Optionally filter out reverse rows
        if drop_reverse and (self.reverse is not None):
            keep = self.reverse == 0
            self.center_paths = list(np.array(self.center_paths)[keep])
            self.left_paths   = list(np.array(self.left_paths)[keep])
            self.right_paths  = list(np.array(self.right_paths)[keep])
            self.steering     = self.steering[keep]
            if self.speed is not None:
                self.speed = self.speed[keep]

        # Build sequence start indices
        N = len(self.steering)
        self.starts = list(range(0, N - seq_len + 1, stride))

        # Use a constant dt if we don't have timestamps
        self.dt_value = 1.0 / float(target_fps)

    def __len__(self):
        return len(self.starts)

    def _load_image(self, path: str) -> Image.Image:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return img.copy()

    def _choose_camera(self):
        if self.camera_mode == "center":
            return "center", 0.0
        # random_three: probabilities slightly favor center
        r = random.random()
        if r < 0.6:
            return "center", 0.0
        elif r < 0.8:
            return "left", +self.lr_corr
        else:
            return "right", -self.lr_corr

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len

        cam_sel, corr = self._choose_camera()

        imgs = []
        steers = []
        speeds = []

        # Decide augmentations for the whole sequence
        do_flip = (random.random() < self.flip_prob)
        do_jitter = self.jitter

        for i in range(start, end):
            if cam_sel == "center":
                path = self.center_paths[i]
            elif cam_sel == "left":
                path = self.left_paths[i]
            else:
                path = self.right_paths[i]

            img = self._load_image(path)
            img = crop_and_resize(img)

            if do_jitter:
                img = color_jitter(img, brightness=0.2, contrast=0.2)

            if do_flip:
                img = horizontal_flip(img)

            imgs.append(to_tensor(img))

            # Steering correction for left/right cameras
            s = float(self.steering[i]) + corr
            if do_flip:
                s = -s  # invert steering on horizontal flip
            steers.append([s])

            if self.speed is not None:
                speeds.append([float(self.speed[i])])

        # Stack to tensors: (T, C, H, W), (T, 1), (T, 1)
        img_t = torch.stack(imgs, dim=0)                      # (T,3,66,200)
        steer_t = torch.tensor(steers, dtype=torch.float32)   # (T,1)
        if self.speed is not None:
            speed_t = torch.tensor(speeds, dtype=torch.float32)  # (T,1)
        else:
            speed_t = torch.zeros((self.seq_len, 1), dtype=torch.float32)

        # Constant dt for the whole sequence
        dt = torch.full((self.seq_len,), float(self.dt_value), dtype=torch.float32)

        sample = {
            "images": img_t,
            "steering": steer_t,
            "speed": speed_t,
            "dt": dt,
        }
        return sample

# -----------------------------
# Model
# -----------------------------

class CNNEncoder(nn.Module):
    """NVIDIA-like encoder mapping (3,66,200) -> feature vector."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        # compute flatten dim by passing a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1,3,66,200)
            y = self.conv(dummy)
            flat_dim = y.view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat_dim, 100), nn.ReLU(inplace=True),
            nn.Linear(100, 50), nn.ReLU(inplace=True),
            nn.Linear(50, out_dim),
        )

    def forward(self, x):  # x: (B,T,3,66,200) or (N,3,66,200)
        orig_shape = x.shape
        if x.dim() == 5:
            B,T,C,H,W = x.shape
            x = x.view(B*T, C, H, W)
            y = self.conv(x)
            y = y.view(y.size(0), -1)
            y = self.head(y)
            y = y.view(B, T, -1)  # (B,T,F)
        else:
            y = self.conv(x)
            y = y.view(y.size(0), -1)
            y = self.head(y)
        return y

class LiquidRNN(nn.Module):
    """
    Sequence model: CNN encoder -> Liquid (CfC) or GRU core -> MLP head for steering (and speed).
    If CfC is unavailable, falls back to GRU (still accepts dt as an extra input channel).
    """
    def __init__(self, feature_dim=128, hidden_dim=128, predict_speed=False, use_cfc=_HAS_NCPS):
        super().__init__()
        self.encoder = CNNEncoder(out_dim=feature_dim)
        self.predict_speed = predict_speed
        self.use_cfc = use_cfc

        if self.use_cfc:
            # CfC cell; we'll unroll it manually over time and concatenate dt to the input.
            self.cfc = CfCCell(input_size=feature_dim + 1, hidden_size=hidden_dim)  # +1 for dt
        else:
            # GRU that consumes [features; dt] as input
            self.gru = nn.GRU(input_size=feature_dim + 1, hidden_size=hidden_dim, batch_first=True)

        # Heads
        self.head_steer = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        if self.predict_speed:
            self.head_speed = nn.Sequential(
                nn.Linear(hidden_dim, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 1)
            )

    def forward(self, images: torch.Tensor, dt: torch.Tensor):
        """
        images: (B,T,3,66,200)
        dt:     (B,T) or (T,) -> we broadcast to (B,T,1)
        returns steering_seq: (B,T,1), speed_seq: (B,T,1 or 0)
        """
        B,T = images.shape[0], images.shape[1]
        feats = self.encoder(images)  # (B,T,F)

        if dt.dim() == 1:
            dt = dt.unsqueeze(0).expand(B, -1)
        dt_in = dt.unsqueeze(-1)      # (B,T,1)

        x = torch.cat([feats, dt_in], dim=-1)  # (B,T,F+1)

        if self.use_cfc:
            # Manual unroll CfC
            h = torch.zeros(B, self.cfc.hidden_size, device=images.device)
            hs = []
            for t in range(T):
                h = self.cfc(x[:, t, :], h, timespans=dt[:, t])
                hs.append(h)
            h_seq = torch.stack(hs, dim=1)  # (B,T,H)
        else:
            h_seq, _ = self.gru(x)  # (B,T,H)

        steer = self.head_steer(h_seq)  # (B,T,1)
        if self.predict_speed:
            speed = self.head_speed(h_seq)  # (B,T,1)
        else:
            speed = None
        return steer, speed

# -----------------------------
# Training / Evaluation
# -----------------------------

def train_epoch(model, loader, optimizer, device, predict_speed=False, lambda_speed=0.1, lambda_smooth=1e-3):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    prev_out = None

    for batch in loader:
        imgs = batch["images"].to(device)         # (B,T,3,66,200)
        dt = batch["dt"].to(device)               # (B,T)
        steer_gt = batch["steering"].to(device)   # (B,T,1)
        speed_gt = batch["speed"].to(device)      # (B,T,1)

        optimizer.zero_grad()
        steer_pred, speed_pred = model(imgs, dt)

        loss = F.mse_loss(steer_pred, steer_gt)

        if predict_speed and (speed_pred is not None):
            loss = loss + lambda_speed * F.mse_loss(speed_pred, speed_gt)

        # Smoothness penalty on output changes (jerk proxy)
        smooth = (steer_pred[:, 1:, :] - steer_pred[:, :-1, :]).pow(2).mean()
        loss = loss + lambda_smooth * smooth

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae = (steer_pred - steer_gt).abs().mean().item()
        total_loss += loss.item() * imgs.size(0)
        total_mae += mae * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n, total_mae / n

@torch.no_grad()
def eval_epoch(model, loader, device, predict_speed=False):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    for batch in loader:
        imgs = batch["images"].to(device)
        dt = batch["dt"].to(device)
        steer_gt = batch["steering"].to(device)
        speed_gt = batch["speed"].to(device)

        steer_pred, speed_pred = model(imgs, dt)
        loss = F.mse_loss(steer_pred, steer_gt)
        mae = (steer_pred - steer_gt).abs().mean().item()

        total_loss += loss.item() * imgs.size(0)
        total_mae  += mae * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n, total_mae / n

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Liquid Neural Network (CfC) on Udacity Behavioral Cloning")
    parser.add_argument("--data_dir", type=str, required=True, help="Root folder that contains IMG/ and driving CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to driving_log.csv (or your CSV)")
    parser.add_argument("--seq_len", type=int, default=20, help="Sequence length (timesteps)")
    parser.add_argument("--stride", type=int, default=1, help="Stride for sliding window")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--camera_mode", type=str, default="random_three", choices=["center","random_three"])
    parser.add_argument("--lr_steer_corr", type=float, default=0.2, help="Steering correction for left/right cameras")
    parser.add_argument("--predict_speed", action="store_true", help="Also predict speed (if available)")
    parser.add_argument("--no_jitter", action="store_true", help="Disable color jitter")
    parser.add_argument("--no_flip", action="store_true", help="Disable horizontal flip")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--target_fps", type=float, default=10.0, help="Assumed fps to build constant dt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="lnn_udacity.pt")

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    if _HAS_NCPS:
        print("ncps detected: using CfC (Liquid) core ✅")
    else:
        print("ncps NOT detected: falling back to GRU core (install with `pip install ncps`) ⚠️")

    dataset = UdacitySeqDataset(
        csv_path=args.csv,
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        camera_mode=args.camera_mode,
        lr_steer_correction=args.lr_steer_corr,
        use_speed_label=args.predict_speed,
        jitter=not args.no_jitter,
        flip_prob=0.0 if args.no_flip else 0.5,
        target_fps=args.target_fps
    )

    # Split
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = LiquidRNN(
        feature_dim=128,
        hidden_dim=128,
        predict_speed=args.predict_speed,
        use_cfc=_HAS_NCPS
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mae = train_epoch(model, train_loader, optimizer, args.device, predict_speed=args.predict_speed)
        va_loss, va_mae = eval_epoch(model, val_loader, args.device, predict_speed=args.predict_speed)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.5f} mae {tr_mae:.5f} | val loss {va_loss:.5f} mae {va_mae:.5f}")

        if va_mae < best_val_mae:
            best_val_mae = va_mae
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_val_mae": best_val_mae
            }, args.save_path)
            print(f"  ✔ Saved new best to {args.save_path} (val MAE {best_val_mae:.5f})")

    print("Training finished.")

if __name__ == "__main__":
    main()
