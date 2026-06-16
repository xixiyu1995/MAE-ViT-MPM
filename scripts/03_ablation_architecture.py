"""
Architecture ablation: sweep over window size, number of heads, and depth.
Transfer strategy (freeze vs fine-tune) is fixed per window size based on Section 4.1:
- ws=8,16: freeze_encoder=True
- ws=32: freeze_encoder=False (no pretraining)
Detailed per-fold results are saved for each deposit.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools
import numpy as np
import pandas as pd
import torch
import time
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from models import MAEEncoder, MAEClassifier
from data_utils import (
    build_grid_data, get_valid_windows, window_contains_point,
    get_deposit_grid_inds, get_window_cells, WindowDatasetWithLabels as WindowDataset
)
from train_utils import finetune_fold

# ==================== Configuration ====================
CSV_PATH = '../data/your_data.csv'
MINERAL_PATH = '../data/mineral_points.csv'
PRETRAIN_DIR = '../outputs/pretrained_models'
OUTPUT_DIR = '../outputs/ablation_results/architecture_ablation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS_FINETUNE = 30
BATCH_SIZE = 128
LR = 1e-4
N_REPEATS = 5
SEED = 42
STEP = 2
MAX_TRANS = 0.4

ELEMENT_COLS = ['AG', 'AS_', 'AU', 'B', 'BA', 'BI', 'CU', 'F', 'HG', 'MN', 'MO', 'PB', 'SB', 'SN', 'W', 'ZN']

PATCH_SIZE = 4
EMBED_DIM = 128          # Fixed embed_dim, all tested heads (4,8,16) divide it evenly
DIM_FEEDFORWARD = 2048
ACTIVATION = 'gelu'
ENCODER_DROPOUT = 0.1    # Dropout for encoder (not swept here)

WINDOW_SIZES = [8, 16, 32]
NUM_HEADS_VALUES = [4, 8, 16]
DEPTH_VALUES = [4, 6, 8]


def run_architecture_ablation():
    grid, mask, x_vals, y_vals = build_grid_data(CSV_PATH, ELEMENT_COLS)
    ny, nx = grid.shape[1], grid.shape[2]
    deposit_inds = get_deposit_grid_inds(MINERAL_PATH, x_vals, y_vals)
    n_deposits = len(deposit_inds)
    print(f"Number of deposits: {n_deposits}")

    for ws in WINDOW_SIZES:
        if ws in [8, 16]:
            freeze_encoder = True
            strategy = "freeze"
        else:  # ws == 32
            freeze_encoder = False
            strategy = "fine-tune (no pretraining)"

        print(f"\n========== Window size: {ws} (transfer strategy: {strategy}) ==========")
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
        print(f"Positive windows: {len(positive_windows)}, Negative windows: {len(negative_windows)}")

        results_ws = []
        combinations = list(itertools.product(NUM_HEADS_VALUES, DEPTH_VALUES))

        for (heads, depth) in tqdm(combinations, desc=f"Architecture ws={ws}"):
            print(f"\n---> heads={heads}, depth={depth}")
            start_time = time.time()

            # All tested heads divide embed_dim=128, so no adjustment needed.
            # If you ever add heads that do not divide 128, you should skip that combination,
            # not modify embed_dim, because pretrained weights are fixed at 128.
            encoder_params = {
                'in_chans': len(ELEMENT_COLS),
                'img_size': ws,
                'patch_size': PATCH_SIZE,
                'embed_dim': EMBED_DIM,
                'depth': depth,
                'num_heads': heads,
                'dim_feedforward': DIM_FEEDFORWARD,
                'activation': ACTIVATION,
                'dropout': ENCODER_DROPOUT
            }

            # Load pretrained encoder if ws != 32
            encoder_state = None
            if ws != 32:
                pretrain_path = os.path.join(MODEL_DIR, f'mae_pretrained_ws{ws}_d{depth}_h{heads}.pth')
                if os.path.exists(pretrain_path):
                    full_state = torch.load(pretrain_path, map_location='cpu')
                    encoder_state = {k: v for k, v in full_state.items()
                                     if k.startswith('patch_embed.') or k.startswith('encoder.') or k == 'pos_embed'}
                else:
                    print(f"Pretrained model missing: {pretrain_path}, skip this combination.")
                    continue

            metrics = {'precision': [], 'recall': [], 'f1': [], 'auc_roc': [], 'auc_pr': []}
            detailed_records = []

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

                    val_neg_cells = set()
                    for w in val_neg:
                        val_neg_cells.update(get_window_cells(w[0], w[1], ws))
                    val_region_cells = val_pos_cells.union(val_neg_cells)

                    # Training negative windows (exclude val_neg, no overlap with val_region)
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

                    probs, trues = finetune_fold(
                        train_pos + train_neg,
                        [1] * len(train_pos) + [0] * len(train_neg),
                        val_pos + val_neg,
                        [1] * len(val_pos) + [0] * len(val_neg),
                        grid, ws, ny, nx,
                        encoder_state, encoder_params,
                        freeze_encoder=freeze_encoder,
                        max_trans=MAX_TRANS,
                        forbidden_cells=val_region_cells,
                        batch_size=BATCH_SIZE,
                        lr=LR,
                        epochs=EPOCHS_FINETUNE,
                        dropout_rate=ENCODER_DROPOUT   # Use same dropout as encoder for classifier head
                    )

                    if len(probs) == 0 or len(np.unique(trues)) < 2:
                        continue

                    pred = (probs >= 0.5).astype(int)
                    prec = precision_score(trues, pred, zero_division=0)
                    rec = recall_score(trues, pred, zero_division=0)
                    f1 = f1_score(trues, pred, zero_division=0)
                    auc_roc = roc_auc_score(trues, probs)
                    prec_curve, rec_curve, _ = precision_recall_curve(trues, probs)
                    auc_pr = auc(rec_curve, prec_curve)

                    metrics['precision'].append(prec)
                    metrics['recall'].append(rec)
                    metrics['f1'].append(f1)
                    metrics['auc_roc'].append(auc_roc)
                    metrics['auc_pr'].append(auc_pr)

                    detailed_records.append({
                        'window_size': ws, 'num_heads': heads, 'depth': depth,
                        'freeze_encoder': freeze_encoder, 'repeat': repeat + 1,
                        'deposit_idx': dep_idx + 1, 'precision': prec, 'recall': rec,
                        'f1': f1, 'auc_roc': auc_roc, 'auc_pr': auc_pr,
                        'n_train_pos': len(train_pos), 'n_train_neg': n_neg_needed,
                        'n_val_pos': len(val_pos), 'n_val_neg': n_val_neg
                    })

            if detailed_records:
                detail_df = pd.DataFrame(detailed_records)
                detail_path = os.path.join(OUTPUT_DIR, f'details_ws{ws}_h{heads}_d{depth}.csv')
                detail_df.to_csv(detail_path, index=False, float_format='%.6f')

            if not metrics['precision']:
                print("No valid results.")
                continue

            def mean_std(lst):
                return np.mean(lst), np.std(lst)

            prec_m, prec_s = mean_std(metrics['precision'])
            rec_m, rec_s = mean_std(metrics['recall'])
            f1_m, f1_s = mean_std(metrics['f1'])
            auc_roc_m, auc_roc_s = mean_std(metrics['auc_roc'])
            auc_pr_m, auc_pr_s = mean_std(metrics['auc_pr'])
            elapsed = time.time() - start_time

            print(f"  F1={f1_m:.4f}±{f1_s:.4f} | AUC-PR={auc_pr_m:.4f}±{auc_pr_s:.4f} | Time={elapsed:.1f}s")

            results_ws.append({
                'window_size': ws, 'num_heads': heads, 'depth': depth,
                'freeze_encoder': freeze_encoder,
                'precision_mean': prec_m, 'precision_std': prec_s,
                'recall_mean': rec_m, 'recall_std': rec_s,
                'f1_mean': f1_m, 'f1_std': f1_s,
                'auc_roc_mean': auc_roc_m, 'auc_roc_std': auc_roc_s,
                'auc_pr_mean': auc_pr_m, 'auc_pr_std': auc_pr_s,
                'n_valid_evals': len(metrics['precision'])
            })

        if results_ws:
            df_ws = pd.DataFrame(results_ws)
            out_path = os.path.join(OUTPUT_DIR, f'summary_architecture_ws{ws}.csv')
            df_ws.to_csv(out_path, index=False, float_format='%.4f')
            print(f"Saved to {out_path}")


if __name__ == '__main__':
    run_architecture_ablation()