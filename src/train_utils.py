"""
Training utilities: finetune_fold, evaluation metrics, etc.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from .data_utils import WindowDatasetWithLabels as WindowDataset  # 带标签版本，重命名以便兼容原脚本


def finetune_fold(train_windows, train_labels, val_windows, val_labels,
                  grid, ws, ny, nx, encoder_state, encoder_params,
                  freeze_encoder, max_trans, forbidden_cells,
                  batch_size=128, lr=1e-4, epochs=30, dropout_rate=0.2):
    """
    Fine-tune a classifier on one fold (train/val split).

    Parameters:
    - train_windows, train_labels: training windows and labels
    - val_windows, val_labels: validation windows and labels
    - grid: geochemical grid (C, H, W)
    - ws: window size
    - ny, nx: grid dimensions
    - encoder_state: pretrained encoder state dict or None
    - encoder_params: parameters for MAEEncoder
    - freeze_encoder: bool, whether to freeze encoder weights
    - max_trans: maximum translation ratio for data augmentation
    - forbidden_cells: set of (y,x) cells that must not be entered during augmentation
    - batch_size, lr, epochs: training hyperparameters
    - dropout_rate: dropout rate for the classifier head (default 0.2)

    Returns:
    - probs: predicted probabilities for validation samples
    - trues: ground truth labels
    """
    from .models import MAEEncoder, MAEClassifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = MAEEncoder(**encoder_params).to(device)
    if encoder_state is not None:
        encoder.load_state_dict(encoder_state, strict=True)

    # 使用传入的 dropout_rate 构建分类器
    model = MAEClassifier(encoder, num_classes=2, dropout_rate=dropout_rate, freeze_encoder=freeze_encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = WindowDataset(train_windows, train_labels, grid, ws, ny, nx,
                             augment=True, max_trans=max_trans, forbidden_cells=forbidden_cells)
    val_ds = WindowDataset(val_windows, val_labels, grid, ws, ny, nx, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        return np.array([]), np.array([])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for _ in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            if torch.isnan(loss) or torch.isinf(loss):
                return np.array([]), np.array([])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device))
            probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            trues.extend(labels.numpy())
    return np.array(probs), np.array(trues)


def compute_metrics(probs, trues):
    """Compute precision, recall, f1, auc_roc, auc_pr from probabilities and ground truth."""
    pred = (probs >= 0.5).astype(int)
    prec = precision_score(trues, pred, zero_division=0)
    rec = recall_score(trues, pred, zero_division=0)
    f1 = f1_score(trues, pred, zero_division=0)
    auc_roc = roc_auc_score(trues, probs)
    prec_curve, rec_curve, _ = precision_recall_curve(trues, probs)
    auc_pr = auc(rec_curve, prec_curve)
    return prec, rec, f1, auc_roc, auc_pr