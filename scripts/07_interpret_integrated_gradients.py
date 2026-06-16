"""
Integrated Gradients interpretability analysis.
Generates attribution heatmaps, profile curves, and combined comparison figures
for the top few mineral deposits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from captum.attr import IntegratedGradients
from models import MAEEncoder, MAEClassifier
from data_utils import build_grid_data, get_deposit_grid_inds, extract_window_around_point

# ==================== Configuration ====================
DATA_PATH = '../data/your_data.csv'
MINERAL_PATH = '../data/mineral_points.csv'
MODEL_PATH = '../outputs/final_model/final_model_ws16_optimal.pth'
WINDOW_SIZE = 16
ELEMENT_COLS = ['AG', 'AS_', 'AU', 'B', 'BA', 'BI', 'CU', 'F', 'HG', 'MN', 'MO', 'PB', 'SB', 'SN', 'W', 'ZN']
# Pretty element labels for plotting
ELEMENT_LABELS = ['Ag', 'As', 'Au', 'B', 'Ba', 'Bi', 'Cu', 'F', 'Hg', 'Mn', 'Mo', 'Pb', 'Sb', 'Sn', 'W', 'Zn']
IG_STEPS = 50
OUTPUT_DIR = '../outputs/ig_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters (must match final training)
PATCH_SIZE = 4
EMBED_DIM = 128
DEPTH = 4
NUM_HEADS = 4
DIM_FEEDFORWARD = 2048
DROPOUT = 0.3
FREEZE_ENCODER = True   # Only needed for model construction


def load_model():
    """Load the trained final model."""
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
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def plot_profile_comparison(attr, win_orig, point_in_window, element_labels, sorted_idx,
                            deposit_idx, global_ig_max, output_dir):
    """
    Plot profile curves of IG attribution vs raw concentration for each element.
    """
    wy, wx = point_in_window
    num_elements = len(element_labels)
    nrows, ncols = 2, 8   # 2 rows, 8 columns = 16 elements
    fig, axes = plt.subplots(nrows, ncols, figsize=(32, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(num_elements):
        idx = sorted_idx[i]
        elem = element_labels[idx]
        ig_values = attr[idx, wy, :]
        raw_values = win_orig[idx, wy, :]

        # Normalize IG globally (symmetric)
        ig_norm = ig_values / (global_ig_max + 1e-8)

        # Normalize raw concentration to [0,1]
        raw_min, raw_max = raw_values.min(), raw_values.max()
        if raw_max - raw_min > 1e-8:
            raw_norm = (raw_values - raw_min) / (raw_max - raw_min)
        else:
            raw_norm = raw_values

        ax = axes[i]
        ax.plot(ig_norm, 'k-', linewidth=1.5)
        ax.fill_between(range(len(ig_norm)), 0, ig_norm, where=(ig_norm > 0), color='red', alpha=0.3)
        ax.fill_between(range(len(ig_norm)), 0, ig_norm, where=(ig_norm < 0), color='blue', alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(wx, color='black', linestyle=':', linewidth=1)
        ax.plot(raw_norm, 'g--', linewidth=1)
        ax.set_title(f"{elem} (Rank {i+1})", fontsize=12)
        ax.grid(True, alpha=0.5)

        # Hide y-axis labels for non-first columns
        if i % ncols != 0:
            ax.tick_params(labelleft=False)

    # Set shared labels
    for i, ax in enumerate(axes):
        if i // ncols == nrows - 1:
            ax.set_xlabel("Column Index", fontsize=12)
        if i % ncols == 0:
            ax.set_ylabel("Normalized Value", fontsize=12)

    axes[0].legend(['IG (globally norm)', 'Positive', 'Negative', 'Raw Conc. (norm)'],
                   fontsize=7, loc='upper right')
    plt.suptitle(f"IG Attribution vs Raw Concentration (Deposit {deposit_idx+1})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, f"profile_comparison_deposit{deposit_idx+1}.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved profile comparison: {save_path}")


def plot_combined_heatmaps(attr, conc_norm, point_in_window, element_labels, deposit_idx,
                           sort_order_horizontal, sort_order_vertical, output_dir):
    """
    Plot combined heatmaps: horizontal/vertical sections of concentration and IG attribution.
    """
    wy, wx = point_in_window
    num_elements = len(element_labels)
    window_size = attr.shape[1]

    # Extract horizontal and vertical slices
    ig_horiz = attr[:, wy, :]               # shape (num_elements, window_size)
    conc_horiz = conc_norm[:, wy, :]
    ig_vert = attr[:, :, wx]                # shape (num_elements, window_size)
    conc_vert = conc_norm[:, :, wx]

    # Global symmetric normalization for IG
    global_ig_max = max(np.abs(attr).max(), 0.01)
    ig_horiz_norm = ig_horiz / global_ig_max
    ig_vert_norm = ig_vert / global_ig_max

    # Reorder elements based on sorting order
    ig_horiz_sorted = ig_horiz_norm[sort_order_horizontal]
    conc_horiz_sorted = conc_horiz[sort_order_horizontal]
    ig_vert_sorted = ig_vert_norm[sort_order_vertical]
    conc_vert_sorted = conc_vert[sort_order_vertical]

    sorted_elements_horiz = [element_labels[i] for i in sort_order_horizontal]
    sorted_elements_vert = [element_labels[i] for i in sort_order_vertical]

    # Normalize concentration per element to [0,1] (min-max)
    conc_horiz_norm = np.zeros_like(conc_horiz_sorted)
    for i in range(num_elements):
        row = conc_horiz_sorted[i]
        minv, maxv = row.min(), row.max()
        if maxv - minv > 1e-8:
            conc_horiz_norm[i] = (row - minv) / (maxv - minv)
        else:
            conc_horiz_norm[i] = row

    conc_vert_norm = np.zeros_like(conc_vert_sorted)
    for i in range(num_elements):
        col = conc_vert_sorted[i]
        minv, maxv = col.min(), col.max()
        if maxv - minv > 1e-8:
            conc_vert_norm[i] = (col - minv) / (maxv - minv)
        else:
            conc_vert_norm[i] = col

    # Transpose vertical data for imshow (rows = spatial index, columns = elements)
    conc_vert_sorted_T = conc_vert_norm.T
    ig_vert_sorted_T = ig_vert_sorted.T

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: Horizontal concentration
    ax1 = axes[0, 0]
    im1 = ax1.imshow(conc_horiz_norm, cmap='YlGn', aspect='auto', origin='upper', vmin=0, vmax=1)
    ax1.axvline(x=wx, color='cyan', linestyle='--', linewidth=2)
    ax1.set_yticks(np.arange(num_elements))
    ax1.set_yticklabels(sorted_elements_horiz, fontsize=10)
    ax1.set_xlabel('Column Index', fontsize=12)
    ax1.set_title('Horizontal Concentration', fontsize=12)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Top-right: Horizontal IG attribution
    ax2 = axes[0, 1]
    im2 = ax2.imshow(ig_horiz_sorted, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto', origin='upper')
    ax2.axvline(x=wx, color='cyan', linestyle='--', linewidth=2)
    ax2.set_yticks(np.arange(num_elements))
    ax2.set_yticklabels(sorted_elements_horiz, fontsize=10)
    ax2.set_xlabel('Column Index', fontsize=12)
    ax2.set_title('Horizontal IG Attribution', fontsize=12)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Bottom-left: Vertical concentration
    ax3 = axes[1, 0]
    im3 = ax3.imshow(conc_vert_sorted_T, cmap='YlGn', aspect='auto', origin='upper', vmin=0, vmax=1)
    ax3.axhline(y=wy, color='cyan', linestyle='--', linewidth=2)
    ax3.set_xticks(np.arange(num_elements))
    ax3.set_xticklabels(sorted_elements_vert, fontsize=10, rotation=45, ha='right')
    ax3.set_ylabel('Row Index', fontsize=12)
    ax3.set_title('Vertical Concentration', fontsize=12)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Bottom-right: Vertical IG attribution
    ax4 = axes[1, 1]
    im4 = ax4.imshow(ig_vert_sorted_T, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto', origin='upper')
    ax4.axhline(y=wy, color='cyan', linestyle='--', linewidth=2)
    ax4.set_xticks(np.arange(num_elements))
    ax4.set_xticklabels(sorted_elements_vert, fontsize=10, rotation=45, ha='right')
    ax4.set_ylabel('Row Index', fontsize=12)
    ax4.set_title('Vertical IG Attribution', fontsize=12)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    plt.suptitle(f'IG vs Concentration (Deposit {deposit_idx+1})', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'combined_heatmaps_deposit{deposit_idx+1}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined heatmaps: {save_path}")


def main():
    print("Loading data and model...")
    grid_norm, _, x_vals, y_vals = build_grid_data(DATA_PATH, ELEMENT_COLS)
    deposit_inds = get_deposit_grid_inds(MINERAL_PATH, x_vals, y_vals)
    model = load_model()
    ig = IntegratedGradients(model)

    ny, nx = grid_norm.shape[1], grid_norm.shape[2]
    print(f"Grid dimensions: {ny} rows x {nx} cols")
    print(f"Number of deposits: {len(deposit_inds)}")

    # Analyze first 5 deposits (or fewer if less exist)
    num_to_analyze = min(5, len(deposit_inds))
    for dep_idx in range(num_to_analyze):
        iy, ix = deposit_inds[dep_idx]
        print(f"\n===== Deposit {dep_idx+1}: grid coordinates (row={iy}, col={ix}) =====")

        # Extract window around deposit
        win_norm, win_orig, (tl_y, tl_x), (wy, wx) = extract_window_around_point(
            grid_norm, grid_norm, iy, ix, WINDOW_SIZE, ny, nx
        )
        # Note: win_orig here is the same as win_norm because we passed grid_norm twice.
        # For true raw concentrations, you would need the original unnormalized grid.
        # Since we don't have it, we use normalized values as a proxy (still meaningful for pattern comparison).

        input_tensor = torch.tensor(win_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        attributions, delta = ig.attribute(input_tensor, target=1, n_steps=IG_STEPS, return_convergence_delta=True)
        attr = attributions.squeeze().cpu().detach().numpy()
        print(f"IG computed, convergence delta = {delta.mean():.4f}")

        # Rank elements by positive contribution
        pos_contrib = np.maximum(attr, 0).sum(axis=(1, 2))
        sorted_idx = np.argsort(pos_contrib)[::-1]
        print("Top positive contributions:")
        for rank, idx in enumerate(sorted_idx[:5]):
            print(f"  {rank+1}. {ELEMENT_LABELS[idx]}: {pos_contrib[idx]:.4f}")

        # Global IG max for normalization
        global_ig_max = max(np.abs(attr).max(), 0.01)

        # Generate profile comparison plot
        plot_profile_comparison(attr, win_orig, (wy, wx), ELEMENT_LABELS, sorted_idx,
                                dep_idx, global_ig_max, OUTPUT_DIR)

        # Compute sorting orders for combined heatmaps
        # Horizontal: sort by peak position then total positive contribution
        ig_prof_horiz = np.maximum(attr[:, wy, :], 0)
        ig_peak_horiz = np.argmax(ig_prof_horiz, axis=1)
        ig_sum_horiz = np.sum(ig_prof_horiz, axis=1)
        sort_horiz = np.lexsort((-ig_sum_horiz, ig_peak_horiz))

        # Vertical: sort by peak row then total positive contribution
        ig_prof_vert = np.maximum(attr[:, :, wx], 0)
        ig_peak_vert = np.argmax(ig_prof_vert, axis=1)
        ig_sum_vert = np.sum(ig_prof_vert, axis=1)
        sort_vert = np.lexsort((-ig_sum_vert, ig_peak_vert))

        # Generate combined heatmaps
        plot_combined_heatmaps(attr, win_norm, (wy, wx), ELEMENT_LABELS, dep_idx,
                               sort_horiz, sort_vert, OUTPUT_DIR)

    print("\nAll interpretability figures generated.")


if __name__ == '__main__':
    main()