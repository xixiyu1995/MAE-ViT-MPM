"""
Train final optimal model on all known deposits with strict spatial isolation.
Generate probability map for the entire study area.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.interpolate import griddata
from models import MAEEncoder, MAEClassifier
from data_utils import (
    build_grid_data, get_valid_windows, window_contains_point,
    get_deposit_grid_inds, get_window_cells, WindowDatasetWithLabels as WindowDataset
)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ==================== Config ====================
CSV_PATH = '../data/your_data.csv'
MINERAL_PATH = '../data/mineral_points.csv'
PRETRAIN_PATH = '../outputs/pretrained_models/mae_pretrained_ws16_d4_h4.pth'
OUTPUT_DIR = '../outputs/final_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 16
TRAIN_STEP = 2
INFER_STEP = 2
NUM_HEADS = 4
DEPTH = 4
DIM_FF = 2048
DROPOUT = 0.3
MAX_TRANS = 0.4
FREEZE_ENCODER = True

EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-4
ELEMENT_COLS = ['AG', 'AS_', 'AU', 'B', 'BA', 'BI', 'CU', 'F', 'HG', 'MN', 'MO', 'PB', 'SB', 'SN', 'W', 'ZN']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Main ====================
def train_and_predict():
    print("\n" + "=" * 60)
    print("Training final model with spatial isolation")
    print("=" * 60)

    # Load data
    grid, mask, x_vals, y_vals = build_grid_data(CSV_PATH, ELEMENT_COLS)
    ny, nx = grid.shape[1], grid.shape[2]
    deposit_inds = get_deposit_grid_inds(MINERAL_PATH, x_vals, y_vals)
    print(f"Deposits: {len(deposit_inds)}")

    all_windows = get_valid_windows(mask, WINDOW_SIZE, TRAIN_STEP)
    if not all_windows:
        raise ValueError("No valid windows")

    # Positive windows: all windows containing any deposit
    positive_set = set()
    for iy, ix in deposit_inds:
        for w in all_windows:
            if window_contains_point(w[0], w[1], WINDOW_SIZE, iy, ix):
                positive_set.add(w)
    train_pos = list(positive_set)
    positive_cells = set()
    for (ty, tx) in train_pos:
        positive_cells.update(get_window_cells(ty, tx, WINDOW_SIZE))
    print(f"Positive windows: {len(train_pos)}")

    # Negative windows: strictly no overlap with positive cells
    candidate_neg = []
    for w in all_windows:
        if w in positive_set:
            continue
        if not get_window_cells(w[0], w[1], WINDOW_SIZE).intersection(positive_cells):
            candidate_neg.append(w)
    print(f"Strict negative candidates: {len(candidate_neg)}")
    if len(candidate_neg) == 0:
        raise ValueError("No negative windows satisfying spatial isolation")

    n_neg = min(len(train_pos) * 4, len(candidate_neg))
    train_neg_idx = np.random.choice(len(candidate_neg), n_neg, replace=False)
    train_neg = [candidate_neg[i] for i in train_neg_idx]
    print(f"Training negatives: {n_neg} (ratio 1:{n_neg / len(train_pos):.1f})")

    # Load pretrained encoder
    if os.path.exists(PRETRAIN_PATH):
        full_state = torch.load(PRETRAIN_PATH, map_location='cpu')
        encoder_state = {k: v for k, v in full_state.items()
                         if k.startswith('patch_embed.') or k.startswith('encoder.') or k == 'pos_embed'}
        print("Loaded pretrained encoder")
    else:
        raise FileNotFoundError(f"Pretrained model not found: {PRETRAIN_PATH}")

    encoder = MAEEncoder(in_chans=len(ELEMENT_COLS), img_size=WINDOW_SIZE, patch_size=4,
                         embed_dim=128, depth=DEPTH, num_heads=NUM_HEADS,
                         dim_feedforward=DIM_FF, dropout=DROPOUT).to(DEVICE)
    encoder.load_state_dict(encoder_state, strict=True)

    model = MAEClassifier(encoder, num_classes=2, dropout_rate=DROPOUT, freeze_encoder=FREEZE_ENCODER).to(DEVICE)
    print(f"Freeze encoder: {FREEZE_ENCODER}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Dataset with augmentation, forbidding positive cells
    train_ds = WindowDataset(train_pos + train_neg,
                             [1] * len(train_pos) + [0] * len(train_neg),
                             grid, WINDOW_SIZE, ny, nx,
                             augment=True, max_trans=MAX_TRANS, forbidden_cells=positive_cells)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Training
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")

    # Save final model
    model_save_path = os.path.join(OUTPUT_DIR, 'final_model_ws16_optimal.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # Inference on entire area
    print("\nGenerating probability map...")
    infer_windows = get_valid_windows(mask, WINDOW_SIZE, INFER_STEP)
    print(f"Inference windows: {len(infer_windows)}")
    model.eval()
    # Use dummy labels (0) for inference dataset; augmentation disabled
    infer_ds = WindowDataset(infer_windows, [0] * len(infer_windows), grid, WINDOW_SIZE, ny, nx, augment=False)
    infer_loader = DataLoader(infer_ds, batch_size=BATCH_SIZE, shuffle=False)

    window_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(infer_loader, desc="Inference"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            window_probs.extend(probs)

    prob_map = np.full((ny, nx), np.nan, dtype=np.float32)
    half = WINDOW_SIZE // 2
    for (tl_y, tl_x), prob in zip(infer_windows, window_probs):
        cy, cx = tl_y + half, tl_x + half
        if 0 <= cy < ny and 0 <= cx < nx:
            prob_map[cy, cx] = prob

    # Interpolation
    valid_mask = ~np.isnan(prob_map)
    if valid_mask.sum() >= 4:
        y_coords, x_coords = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        points = np.column_stack((x_coords[valid_mask], y_coords[valid_mask]))
        values = prob_map[valid_mask]
        xi, yi = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
        prob_map_interp = griddata(points, values, (xi, yi), method='cubic')
        prob_map = np.clip(prob_map_interp, 0, 1)
        prob_map[~mask] = np.nan
    else:
        prob_map[~mask] = np.nan

    np.save(os.path.join(OUTPUT_DIR, 'prob_map.npy'), prob_map)
    yy, xx = np.meshgrid(y_vals, x_vals, indexing='ij')
    df_prob = pd.DataFrame({'POINT_X': xx.flatten(), 'POINT_Y': yy.flatten(), 'PROB': prob_map.flatten()})
    df_prob = df_prob.dropna()
    df_prob.to_csv(os.path.join(OUTPUT_DIR, 'prob_map.csv'), index=False, float_format='%.6f')
    print(f"Probability map saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    train_and_predict()