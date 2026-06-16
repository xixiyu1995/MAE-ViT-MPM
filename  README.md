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


## 🔧 Installation

```bash
git clone https://github.com/yourname/Geochemistry-MAE-Mineral-Prospecting.git
cd Geochemistry-MAE-Mineral-Prospecting
pip install -r requirements.txt

📊 Data Preparation

Place two CSV files in data/:

    your_data.csv – Interpolated geochemical grid.
    Columns: POINT_X, POINT_Y, and one column per element (e.g., AG, AS_, AU, …).
    Grid must be regular (constant Δx, Δy). Values will be Z‑score normalized per element.

    mineral_points.csv – Known deposit coordinates.
    Columns: POINT_X, POINT_Y (must match the grid coordinate system).

All scripts use paths relative to the project root (data/your_data.csv, outputs/...).
Important: The data/ and outputs/ folders are ignored by git (see .gitignore).

🚀 Running the Pipeline

Always run scripts from the project root directory. Example:
bash

python scripts/01_pretrain_mae.py

1. Pretrain MAE models (01_pretrain_mae.py)

Trains MAE for all combinations of window sizes (8,16,32), depths (4,6,8) and heads (4,8,16).
Pretrained weights saved to outputs/pretrained_models/.
2. Ablation: pretraining effectiveness (02_ablation_finetune.py)

Compares with/without pretraining and freeze vs. fine‑tune.
Results: outputs/ablation_results/finetune_ablation/ablation_finetune.csv
3. Architecture ablation (03_ablation_architecture.py)

Sweeps heads and depth while fixing best transfer strategy per window size.
Detailed folds: outputs/ablation_results/architecture_ablation/
4. Training hyperparameter ablation (04_ablation_training.py)

Sweeps dropout (0.1–0.3) and translation range (0.2–0.6) with optimal architecture.
5. Train final model (05_train_final_model.py)

Uses all known deposits as positives, strict negative sampling (no grid cell overlap with any positive window).
Outputs:

    outputs/final_model/final_model_ws16_optimal.pth

    outputs/final_model/prob_map.npy and prob_map.csv (continuous prospectivity map)

6. Monte Carlo Dropout uncertainty (06_mc_dropout_uncertainty.py)

Computes mean probability and epistemic uncertainty (std) maps.
Outputs saved in outputs/Probability_Uncertainty/.
7. Integrated Gradients interpretation (07_interpret_integrated_gradients.py)

Generates attribution heatmaps, profile curves, and combined figures for the first few deposits.
Figures saved in outputs/ig_figures/.
📈 Expected Outputs

    Pretraining logs – Final MAE loss per architecture.

    Ablation CSVs – Mean ± std of precision, recall, F1, AUC‑ROC, AUC‑PR.

    Probability map – Grid of mineralisation potential in [0,1], interpolated via cubic interpolation.

    Uncertainty map – Spatial distribution of model uncertainty (higher near data‑sparse areas).

    IG figures – Element‑wise contributions, showing promoting (positive) vs. inhibiting (negative) geochemical signatures.


