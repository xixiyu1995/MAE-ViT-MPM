"""
Monte Carlo Dropout uncertainty estimation.
Automatically infers model architecture from the final checkpoint,
generates mean probability map and epistemic uncertainty (standard deviation) map.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from models import MAEEncoder, MAEClassifier
from data_utils import build_grid_data

# ==================== Configuration ====================
MODEL_PATH = '../outputs/final_model/final_model_ws16_optimal.pth'
OUTPUT_DIR = '../outputs/Probability_Uncertainty'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_CSV = '../data/your_data.csv'
ELEMENT_COLS = ['AG', 'AS_', 'AU', 'B', 'BA', 'BI', 'CU', 'F', 'HG', 'MN', 'MO', 'PB', 'SB', 'SN', 'W', 'ZN']
STEP = 2
MC_SAMPLES = 30
BATCH_SIZE = 128

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Final model hyperparameters (must match training)
WINDOW_SIZE = 16
PATCH_SIZE = 4
EMBED_DIM = 128
DEPTH = 4
NUM_HEADS = 4
DIM_FEEDFORWARD = 2048
DROPOUT = 0.3
FREEZE_ENCODER = True  # Not used during inference, but required for model construction


class AllWindowDataset(Dataset):
    """Dataset for inference: extracts all valid sliding windows without labels."""
    def __init__(self, grid, mask, window_size, step):
        self.grid = grid
        self.window_size = window_size
        ny, nx = mask.shape
        self.windows = []
        for y0 in range(0, ny - window_size + 1, step):
            for x0 in range(0, nx - window_size + 1, step):
                if mask[y0:y0 + window_size, x0:x0 + window_size].all():
                    w = self.grid[:, y0:y0 + window_size, x0:x0 + window_size]
                    if not np.isnan(w).any():
                        self.windows.append((y0, x0))
        print(f"Number of valid inference windows: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        y0, x0 = self.windows[idx]
        window = self.grid[:, y0:y0 + self.window_size, x0:x0 + self.window_size]
        return torch.tensor(window, dtype=torch.float32), y0, x0


def enable_dropout(m):
    """Set dropout layers to training mode."""
    if isinstance(m, nn.Dropout):
        m.train()


def main():
    print("Loading data...")
    grid, mask, x_vals, y_vals = build_grid_data(DATA_CSV, ELEMENT_COLS)
    ny, nx = grid.shape[1], grid.shape[2]
    print(f"Grid size: {nx} x {ny}")

    print("Building model...")
    encoder = MAEEncoder(
        in_chans=len(ELEMENT_COLS),
        img_size=WINDOW_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    model = MAEClassifier(
        encoder,
        num_classes=2,
        dropout_rate=DROPOUT,
        freeze_encoder=FREEZE_ENCODER
    ).to(DEVICE)

    # Load final model weights
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    model.apply(enable_dropout)   # Enable dropout during inference

    dataset = AllWindowDataset(grid, mask, WINDOW_SIZE, STEP)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    prob_mean = np.full((ny, nx), np.nan, dtype=np.float32)
    prob_std = np.full((ny, nx), np.nan, dtype=np.float32)

    print(f"Running Monte Carlo Dropout (T={MC_SAMPLES})...")
    for batch in tqdm(loader):
        imgs, y0s, x0s = batch
        imgs = imgs.to(DEVICE)
        B = imgs.shape[0]
        all_probs = torch.zeros(MC_SAMPLES, B, 2).to(DEVICE)

        with torch.no_grad():
            for t in range(MC_SAMPLES):
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                all_probs[t] = probs

        prob_class1 = all_probs[:, :, 1]   # probability of class 1 (mineralized)
        mean_prob = prob_class1.mean(dim=0).cpu().numpy()
        std_prob = prob_class1.std(dim=0).cpu().numpy()

        half = WINDOW_SIZE // 2
        for i in range(B):
            cy = y0s[i].item() + half
            cx = x0s[i].item() + half
            if 0 <= cy < ny and 0 <= cx < nx:
                prob_mean[cy, cx] = mean_prob[i]
                prob_std[cy, cx] = std_prob[i]

    # Interpolation to fill missing cells
    valid_mean = ~np.isnan(prob_mean)
    if valid_mean.sum() >= 4:
        rows, cols = np.indices((ny, nx))
        points = np.column_stack((cols[valid_mean], rows[valid_mean]))
        values_mean = prob_mean[valid_mean]
        xi, yi = np.meshgrid(np.arange(nx), np.arange(ny))
        prob_mean_interp = griddata(points, values_mean, (xi, yi), method='cubic')
        prob_mean = np.clip(prob_mean_interp, 0, 1)
        prob_mean[~mask] = np.nan

    valid_std = ~np.isnan(prob_std)
    if valid_std.sum() >= 4:
        rows, cols = np.indices((ny, nx))
        points = np.column_stack((cols[valid_std], rows[valid_std]))
        values_std = prob_std[valid_std]
        prob_std_interp = griddata(points, values_std, (xi, yi), method='cubic')
        prob_std = prob_std_interp   # No clipping for std (can be >1)
        prob_std[~mask] = np.nan

    # Save arrays
    np.save(os.path.join(OUTPUT_DIR, 'prob_mean.npy'), prob_mean)
    np.save(os.path.join(OUTPUT_DIR, 'prob_std.npy'), prob_std)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im1 = axes[0].imshow(prob_mean, cmap='viridis', origin='lower', vmin=0, vmax=1)
    axes[0].set_title('Posterior Mean Probability')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    vmax_std = np.nanpercentile(prob_std, 98) if np.any(~np.isnan(prob_std)) else 0.5
    im2 = axes[1].imshow(prob_std, cmap='RdYlGn_r', origin='lower', vmax=vmax_std)
    axes[1].set_title('Epistemic Uncertainty (Std)')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(f'MC Dropout Uncertainty (T={MC_SAMPLES})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'uncertainty_map.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # Save to CSV
    yy, xx = np.meshgrid(y_vals, x_vals, indexing='ij')
    df = pd.DataFrame({
        'POINT_X': xx.flatten(),
        'POINT_Y': yy.flatten(),
        'MEAN': prob_mean.flatten(),
        'STD': prob_std.flatten()
    }).dropna()
    df.to_csv(os.path.join(OUTPUT_DIR, 'uncertainty_points.csv'), index=False, float_format='%.6f')

    print(f"Mean uncertainty (std) after interpolation: {np.nanmean(prob_std):.4f}")
    print(f"All results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()