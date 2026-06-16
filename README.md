# Geochemistry MAE for Mineral Prospectivity Mapping

This repository implements a **Masked Autoencoder (MAE)** based on Vision Transformer for geochemical anomaly detection and mineral prospectivity mapping. The framework uses strict spatial isolation to avoid data leakage and includes comprehensive ablation studies, final prediction with uncertainty quantification, and model interpretation via Integrated Gradients.

## 📁 Project Structure

├── data/ # Place your CSV data here (not tracked by git)
├── outputs/ # All results, models, and figures (auto‑created)
├── scripts/ # Executable scripts (run in order)
│ ├── 01_pretrain_mae.py
│ ├── 02_ablation_finetune.py
│ ├── 03_ablation_architecture.py
│ ├── 04_ablation_training.py
│ ├── 05_train_final_model.py
│ ├── 06_mc_dropout_uncertainty.py
│ └── 07_interpret_integrated_gradients.py
├── src/ # Core modules
│ ├── data_utils.py
│ ├── models.py
│ ├── train_utils.py
│ └── init.py
├── .gitignore
├── README.md
└── requirements.txt
