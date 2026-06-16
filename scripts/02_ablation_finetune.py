"""
Ablation study: pretraining effectiveness (with/without pretraining) and transfer strategy (freeze vs fine-tune)
Window sizes: 8, 16, 32 (64 can be added if needed)
Repeats: 5, fine-tune epochs: 30
Strict spatial isolation: train/val windows share no grid cells, augmentation avoids validation region.
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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
import warnings
from tqdm import tqdm
import time
from models import MAEEncoder, MAEClassifier
from data_utils import (
    build_grid_data, get_valid_windows, window_contains_point,
    get_deposit_grid_inds, get_window_cells, WindowDatasetWithLabels as WindowDataset
)
from train_utils import finetune_fold

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
CSV_PATH = '../data/your_data.csv'
MINERAL_PATH = '../data/mineral_points.csv'
PRETRAIN_DIR = '../outputs/pretrained_models'
OUTPUT_DIR = '../outputs/ablation_results/finetune_ablation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS_FINETUNE = 30
BATCH_SIZE = 128
LR = 1e-4
N_REPEATS = 5
SEED = 42

ELEMENT_COLS = ['AG', 'AS_', 'AU', 'B', 'BA', 'BI', 'CU', 'F', 'HG', 'MN', 'MO', 'PB', 'SB', 'SN', 'W', 'ZN']

# Fixed architecture parameters
PATCH_SIZE = 4
EMBED_DIM = 128
DEPTH = 6
NUM_HEADS = 4
DIM_FEEDFORWARD = 2048
ENCODER_DROPOUT = 0.1
ACTIVATION = 'gelu'
MAX_TRANS = 0.4
STEP = 2
WINDOW_SIZES = [8, 16, 32]


def run_ablation():
    grid, mask, x_vals, y_vals = build_grid_data(CSV_PATH, ELEMENT_COLS)
    ny, nx = grid.shape[1], grid.shape[2]
    deposit_inds = get_deposit_grid_inds(MINERAL_PATH, x_vals, y_vals)
    n_deposits = len(deposit_inds)
    print(f"Number of deposits: {n_deposits}")

    results = []
    combinations = []
    for ws in WINDOW_SIZES:
        for pretrain in [True, False]:
            if pretrain:
                for freeze in [True, False]:
                    combinations.append((ws, pretrain, freeze))
            else:
                combinations.append((ws, pretrain, False))

    for (ws, pretrain, freeze) in tqdm(combinations, desc="Ablation Progress"):
        print(f"\n========== ws={ws}, pretrain={pretrain}, freeze={freeze} ==========")
        start_time = time.time()

        all_windows = get_valid_windows(mask, ws, STEP)
        if not all_windows:
            print("No valid windows, skip.")
            continue

        positive_set = set()
        for iy, ix in deposit_inds:
            for w in all_windows:
                if window_contains_point(w[0], w[1], ws, iy, ix):
                    positive_set.add(w)
        positive_windows = list(positive_set)
        negative_windows = list(set(all_windows) - positive_set)
        if not positive_windows:
            print("No positive windows, skip.")
            continue

        encoder_params = {
            'in_chans': len(ELEMENT_COLS),
            'img_size': ws,
            'patch_size': PATCH_SIZE,
            'embed_dim': EMBED_DIM,
            'depth': DEPTH,
            'num_heads': NUM_HEADS,
            'dim_feedforward': DIM_FEEDFORWARD,
            'activation': ACTIVATION,
            'dropout': ENCODER_DROPOUT
        }

        metrics = {'precision': [], 'recall': [], 'f1': [], 'auc_roc': [], 'auc_pr': []}

        for repeat in range(N_REPEATS):
            np.random.seed(SEED + repeat)
            torch.manual_seed(SEED + repeat)

            for dep_idx in range(n_deposits):
                # Validation positive windows
                val_pos = [w for w in all_windows if window_contains_point(w[0], w[1], ws, *deposit_inds[dep_idx])]
                if not val_pos:
                    continue

                val_pos_cells = set()
                for (ty, tx) in val_pos:
                    val_pos_cells.update(get_window_cells(ty, tx, ws))

                # Training positive windows (other deposits, no overlap with val_pos_cells)
                train_pos = []
                for oidx in range(n_deposits):
                    if oidx == dep_idx:
                        continue
                    oy, ox = deposit_inds[oidx]
                    candidates = [w for w in all_windows if window_contains_point(w[0], w[1], ws, oy, ox)]
                    for w in candidates:
                        if not get_window_cells(w[0], w[1], ws).intersection(val_pos_cells):
                            train_pos.append(w)
                train_pos = list(set(train_pos))
                if not train_pos:
                    continue

                train_pos_cells = set()
                for w in train_pos:
                    train_pos_cells.update(get_window_cells(w[0], w[1], ws))

                # Candidate negative windows (no overlap with val_pos_cells)
                candidate_neg_all = [w for w in negative_windows
                                     if not get_window_cells(w[0], w[1], ws).intersection(val_pos_cells)]

                # Validation negative windows (also no overlap with train_pos_cells)
                candidate_val_neg = [w for w in candidate_neg_all
                                     if not get_window_cells(w[0], w[1], ws).intersection(train_pos_cells)]
                if not candidate_val_neg:
                    continue

                n_val_neg = min(len(val_pos), len(candidate_val_neg))
                val_neg_idx = np.random.choice(len(candidate_val_neg), n_val_neg, replace=False)
                val_neg = [candidate_val_neg[i] for i in val_neg_idx]

                # Full validation region
                val_neg_cells = set()
                for w in val_neg:
                    val_neg_cells.update(get_window_cells(w[0], w[1], ws))
                val_region_cells = val_pos_cells.union(val_neg_cells)

                # Training negative windows (exclude those used as val_neg, and no overlap with val_region)
                val_neg_set = set(val_neg)
                candidate_train_neg = []
                for w in candidate_neg_all:
                    if w in val_neg_set:
                        continue
                    if not get_window_cells(w[0], w[1], ws).intersection(val_region_cells):
                        candidate_train_neg.append(w)

                n_neg_needed = min(len(train_pos) * 4, len(candidate_train_neg))
                if n_neg_needed == 0:
                    continue
                train_neg_idx = np.random.choice(len(candidate_train_neg), n_neg_needed, replace=False)
                train_neg = [candidate_train_neg[i] for i in train_neg_idx]

                # Load pretrained weights if needed
                encoder_state = None
                if pretrain:
                    pretrain_path = os.path.join(PRETRAIN_DIR, f'mae_pretrained_ws{ws}_d{DEPTH}_h{NUM_HEADS}.pth')
                    if os.path.exists(pretrain_path):
                        full_state = torch.load(pretrain_path, map_location='cpu')
                        encoder_state = {}
                        for k, v in full_state.items():
                            if k.startswith('patch_embed.') or k.startswith('encoder.') or k == 'pos_embed':
                                encoder_state[k] = v
                    else:
                        print(f"Pretrained model missing: {pretrain_path}, skip this fold.")
                        continue

                probs, trues = finetune_fold(
                    train_pos + train_neg,
                    [1] * len(train_pos) + [0] * len(train_neg),
                    val_pos + val_neg,
                    [1] * len(val_pos) + [0] * len(val_neg),
                    grid, ws, ny, nx,
                    encoder_state, encoder_params,
                    freeze_encoder=freeze,
                    max_trans=MAX_TRANS,
                    forbidden_cells=val_region_cells,
                    batch_size=BATCH_SIZE,
                    lr=LR,
                    epochs=EPOCHS_FINETUNE,
                    dropout_rate=ENCODER_DROPOUT   # Use same dropout as encoder (0.1)
                )

                if len(probs) == 0 or len(np.unique(trues)) < 2:
                    continue

                pred = (probs >= 0.5).astype(int)
                metrics['precision'].append(precision_score(trues, pred, zero_division=0))
                metrics['recall'].append(recall_score(trues, pred, zero_division=0))
                metrics['f1'].append(f1_score(trues, pred, zero_division=0))
                metrics['auc_roc'].append(roc_auc_score(trues, probs))
                prec_curve, rec_curve, _ = precision_recall_curve(trues, probs)
                metrics['auc_pr'].append(auc(rec_curve, prec_curve))

        if not metrics['precision']:
            print("No valid results, skip.")
            continue

        def mean_std(lst):
            return np.mean(lst), np.std(lst)

        prec_m, prec_s = mean_std(metrics['precision'])
        rec_m, rec_s = mean_std(metrics['recall'])
        f1_m, f1_s = mean_std(metrics['f1'])
        auc_roc_m, auc_roc_s = mean_std(metrics['auc_roc'])
        auc_pr_m, auc_pr_s = mean_std(metrics['auc_pr'])
        elapsed = time.time() - start_time

        print(f"  Pre={prec_m:.4f}±{prec_s:.4f}  Rec={rec_m:.4f}±{rec_s:.4f}  F1={f1_m:.4f}±{f1_s:.4f}")
        print(f"  AUC_ROC={auc_roc_m:.4f}±{auc_roc_s:.4f}  AUC_PR={auc_pr_m:.4f}±{auc_pr_s:.4f}   Time={elapsed:.1f}s")

        results.append({
            'window_size': ws,
            'pretrain': pretrain,
            'freeze_encoder': freeze,
            'n_pos_windows': len(positive_windows),
            'n_neg_windows': len(negative_windows),
            'precision_mean': prec_m, 'precision_std': prec_s,
            'recall_mean': rec_m, 'recall_std': rec_s,
            'f1_mean': f1_m, 'f1_std': f1_s,
            'auc_roc_mean': auc_roc_m, 'auc_roc_std': auc_roc_s,
            'auc_pr_mean': auc_pr_m, 'auc_pr_std': auc_pr_s,
        })

    df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, 'ablation_finetune.csv')
    df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"\nAblation complete, results saved to {out_path}")


if __name__ == '__main__':
    run_ablation()