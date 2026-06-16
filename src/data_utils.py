"""
Data loading, grid building, window extraction, spatial isolation helpers

Two dataset classes:
- WindowDataset: for pretraining (no labels, sliding window with optional center-only)
- WindowDatasetWithLabels: for fine-tuning/classification (with labels and augmentation)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def build_grid_data(csv_path, element_cols):
    """Load CSV, build grid (C, H, W), apply Z-score normalization, return grid and mask."""
    data = pd.read_csv(csv_path)
    x_vals = np.sort(data['POINT_X'].unique())
    y_vals = np.sort(data['POINT_Y'].unique())
    nx, ny = len(x_vals), len(y_vals)
    n_elements = len(element_cols)

    grid = np.full((n_elements, ny, nx), np.nan, dtype=np.float32)
    x_to_idx = {x: i for i, x in enumerate(x_vals)}
    y_to_idx = {y: i for i, y in enumerate(y_vals)}
    for _, row in data.iterrows():
        ix = x_to_idx[row['POINT_X']]
        iy = y_to_idx[row['POINT_Y']]
        for j, elem in enumerate(element_cols):
            grid[j, iy, ix] = row[elem]

    # Z-score normalization
    for j in range(n_elements):
        valid = grid[j][~np.isnan(grid[j])]
        if len(valid) > 1:
            mean = valid.mean()
            std = valid.std()
            if std > 1e-6:
                grid[j] = (grid[j] - mean) / std
            else:
                grid[j] = grid[j] - mean
        else:
            grid[j] = 0.0

    mask = ~np.isnan(grid[0])
    return grid, mask, x_vals, y_vals


def get_valid_windows(mask, window_size, step):
    """Return list of (y0, x0) for all windows without NaN."""
    ny, nx = mask.shape
    valid = []
    for y0 in range(0, ny - window_size + 1, step):
        for x0 in range(0, nx - window_size + 1, step):
            if mask[y0:y0+window_size, x0:x0+window_size].all():
                valid.append((y0, x0))
    return valid


def window_contains_point(tl_y, tl_x, win_size, point_iy, point_ix):
    """Check if grid point (iy,ix) lies inside window with top-left (tl_y,tl_x)."""
    return (tl_y <= point_iy < tl_y + win_size) and (tl_x <= point_ix < tl_x + win_size)


def get_deposit_grid_inds(deposit_csv, x_vals, y_vals):
    """Convert deposit coordinates to grid indices (iy, ix)."""
    deposits = pd.read_csv(deposit_csv).dropna(subset=['POINT_X', 'POINT_Y'])
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    x_min, y_min = x_vals.min(), y_vals.min()
    nx, ny = len(x_vals), len(y_vals)
    inds = []
    for x, y in zip(deposits['POINT_X'], deposits['POINT_Y']):
        ix = int(round((x - x_min) / dx))
        iy = int(round((y - y_min) / dy))
        ix = max(0, min(ix, nx-1))
        iy = max(0, min(iy, ny-1))
        inds.append((iy, ix))
    return inds


def get_window_cells(tl_y, tl_x, win_size):
    """Return set of all grid cell coordinates inside window."""
    cells = set()
    for y in range(tl_y, tl_y + win_size):
        for x in range(tl_x, tl_x + win_size):
            cells.add((y, x))
    return cells


def safe_translate_window(grid, tl_y, tl_x, win_size, ny, nx, max_trans, forbidden_cells, max_attempts=10):
    """Translate window, ensuring no overlap with forbidden_cells. Return translated window data."""
    trans = int(win_size * max_trans)
    for _ in range(max_attempts):
        dy = np.random.randint(-trans, trans + 1)
        dx = np.random.randint(-trans, trans + 1)
        new_y = max(0, min(tl_y + dy, ny - win_size))
        new_x = max(0, min(tl_x + dx, nx - win_size))
        overlap = False
        for y in range(new_y, new_y + win_size):
            for x in range(new_x, new_x + win_size):
                if (y, x) in forbidden_cells:
                    overlap = True
                    break
            if overlap:
                break
        if not overlap:
            return grid[:, new_y:new_y+win_size, new_x:new_x+win_size]
    return grid[:, tl_y:tl_y+win_size, tl_x:tl_x+win_size]


def extract_window_around_point(grid_norm, grid_orig, iy, ix, window_size, ny, nx):
    """
    Extract a window around a given grid point, as centered as possible.
    Used in Integrated Gradients interpretation.
    """
    half = window_size // 2
    tl_y = max(0, min(iy - half, ny - window_size))
    tl_x = max(0, min(ix - half, nx - window_size))
    win_norm = grid_norm[:, tl_y:tl_y+window_size, tl_x:tl_x+window_size]
    win_orig = grid_orig[:, tl_y:tl_y+window_size, tl_x:tl_x+window_size]
    wy = iy - tl_y
    wx = ix - tl_x
    return win_norm, win_orig, (tl_y, tl_x), (wy, wx)


# ---------------------------
# Dataset classes
# ---------------------------

class WindowDataset(Dataset):
    """
    Dataset for pretraining (no labels).
    Accepts full grid and mask, extracts all valid sliding windows.
    """
    def __init__(self, grid_data, mask, window_size=32, step=2, center_only=False):
        self.grid = grid_data
        self.mask = mask
        self.window_size = window_size
        self.step = step
        self.center_only = center_only
        _, self.ny, self.nx = grid_data.shape

        y_starts = list(range(0, self.ny - window_size + 1, step))
        x_starts = list(range(0, self.nx - window_size + 1, step))

        self.valid_windows = []
        half = window_size // 2
        for y0 in y_starts:
            for x0 in x_starts:
                if center_only:
                    cy, cx = y0 + half, x0 + half
                    if mask[cy, cx]:
                        window = self.grid[:, y0:y0+window_size, x0:x0+window_size]
                        if not np.isnan(window).any():
                            self.valid_windows.append((y0, x0))
                else:
                    window_mask = mask[y0:y0+window_size, x0:x0+window_size]
                    if window_mask.all():
                        window = self.grid[:, y0:y0+window_size, x0:x0+window_size]
                        if not np.isnan(window).any():
                            self.valid_windows.append((y0, x0))
        self.total = len(self.valid_windows)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        y0, x0 = self.valid_windows[idx]
        window = self.grid[:, y0:y0+self.window_size, x0:x0+self.window_size]
        return torch.tensor(window, dtype=torch.float32)


class WindowDatasetWithLabels(Dataset):
    """
    Dataset for fine-tuning / classification.
    Accepts explicit list of windows (top-left corners) and corresponding labels.
    Supports optional data augmentation (safe translation) for positive samples.
    """
    def __init__(self, windows, labels, grid, win_size, ny, nx,
                 augment=False, max_trans=0.4, forbidden_cells=None):
        self.samples = []
        for (ty, tx), lbl in zip(windows, labels):
            w = grid[:, ty:ty+win_size, tx:tx+win_size]
            if not (np.isnan(w).any() or np.isinf(w).any()):
                self.samples.append((ty, tx, w, lbl))
        self.grid = grid
        self.win_size = win_size
        self.ny, self.nx = ny, nx
        self.augment = augment
        self.max_trans = max_trans
        self.forbidden_cells = forbidden_cells if forbidden_cells is not None else set()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ty, tx, w, lbl = self.samples[idx]
        if self.augment and lbl == 1:
            w = safe_translate_window(self.grid, ty, tx, self.win_size, self.ny, self.nx,
                                      self.max_trans, self.forbidden_cells)
        return torch.tensor(w, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)