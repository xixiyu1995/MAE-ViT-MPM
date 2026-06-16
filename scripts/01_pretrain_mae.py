"""
Pretrain MAE models with different window sizes, depths, heads.
Output: mae_pretrained_ws{w}_d{depth}_h{heads}.pth
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from models import MAE
from data_utils import build_grid_data, WindowDataset

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def train_mae(csv_path, element_cols, save_dir,
              window_size=32, depth=6, num_heads=4,
              step=2, batch_size=256, epochs=200, lr=1.5e-4,
              use_amp=True, num_workers=0, center_only=False):
    """Train a single MAE model with given architecture."""
    grid, mask, _, _ = build_grid_data(csv_path, element_cols)
    dataset = WindowDataset(grid, mask, window_size, step, center_only=center_only)
    if len(dataset) == 0:
        raise ValueError("No valid windows")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=(num_workers > 0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MAE(in_chans=len(element_cols), img_size=window_size, patch_size=4,
                embed_dim=128, depth=depth, num_heads=num_heads,
                decoder_embed_dim=64, decoder_depth=3, decoder_num_heads=4,
                mask_ratio=0.75).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MAE: Depth={depth}, Heads={num_heads} | Params: {total_params / 1e6:.2f}M")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'mae_pretrained_ws{window_size}_d{depth}_h{num_heads}.pth')
    if os.path.exists(save_path):
        print(f"Model {save_path} exists, skip.")
        return

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs} (ws={window_size}, d={depth}, h={num_heads})", leave=False)
        for imgs in pbar:
            imgs = imgs.to(device, non_blocking=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = model(imgs)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(imgs)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(loader)
            tqdm.write(f"  -> Epoch {epoch + 1} Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Pretrained saved to {save_path}\n")


if __name__ == '__main__':
    # Paths relative to project root
    CSV_PATH = '../data/your_data.csv'
    ELEMENT_COLS = ['AG', 'AS_', 'AU', 'B', 'BA', 'BI', 'CU', 'F', 'HG', 'MN', 'MO', 'PB', 'SB', 'SN', 'W', 'ZN']
    SAVE_DIR = '../outputs/pretrained_models'

    window_sizes = [8, 16, 32]
    depths = [4, 6, 8]
    heads = [4, 8, 16]

    combinations = list(itertools.product(window_sizes, depths, heads))
    print(f"Total pretraining tasks: {len(combinations)}")

    for ws, d, h in combinations:
        print(f"\n========== Window={ws}, Depth={d}, Heads={h} ==========")
        batch_size_dyn = 256 if ws <= 16 else 128
        train_mae(CSV_PATH, ELEMENT_COLS, SAVE_DIR,
                  window_size=ws, depth=d, num_heads=h,
                  step=2, batch_size=batch_size_dyn, epochs=200,
                  lr=1.5e-4, use_amp=True, num_workers=0, center_only=False)