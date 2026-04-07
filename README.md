# SD6123 Group Assignment — Privacy-Preserved Machine Learning

This repository contains the full codebase for our group project, covering federated learning algorithm implementations, privacy attack analysis, and privacy defence mechanisms using Differential Privacy (DP).

---

## Repository Structure

```
├── Centralized baseline + 3 FL algorithms/
│   ├── fednova (CIFAR-10)/          # FL experiments on CIFAR-10
│   ├── fednova (CIFAR-100)/         # FL experiments on CIFAR-100
│   └── run_*.sh                     # HPC job scripts (see note below)
│
├── Privacy Attack Code/
│   ├── final_privacy_attacks-3.ipynb  # Privacy attack analysis notebook
│   ├── *_privacy.csv                # Pre-generated privacy metric data
│   └── partition_indices_*.json     # Client data partition indices
│
└── Privacy Defence Code/
    ├── FEDNOVA/                     # FL with Differential Privacy (DP-SGD)
    │   ├── conf/                    # Hydra config files
    │   └── *.py                     # Training, strategy, model files
    └── TC2 JOB SCRIPT/              # HPC job scripts (see note below)
```

---

## Environment Setup

### Requirements

Install all dependencies with:

```bash
pip install torch torchvision flwr hydra-core omegaconf numpy pandas matplotlib seaborn scikit-learn scipy opacus
```

Or install from the requirements file if provided:

```bash
pip install -r requirements.txt
```

> **Python version:** 3.10 recommended (used during development).

### Dataset

**Do not download the dataset manually.** CIFAR-10 and CIFAR-100 are downloaded automatically by PyTorch on first run into the `data/` folder (which is excluded from this repository due to size).

---

## Folder 1 — Centralized Baseline + 3 FL Algorithms 

Contains implementations of:
- **Centralized** baseline training
- **FedAvg** — Federated Averaging
- **FedProx** — Federated Proximal
- **FedNova** — Federated Nova-Proximal

Adapted from:
<ins>https://github.com/flwrlabs/flower/tree/main/baselines</ins>

Both CIFAR-10 and CIFAR-100 variants are included.

### Configuration

All experiment settings are controlled via `conf/base.yaml`:

| Parameter | Description |
|---|---|
| `datapath` | Path to dataset (auto-created on first run) |
| `checkpoint_path` | Where model checkpoints are saved |
| `num_clients` | Number of federated clients (default: 16) |
| `num_rounds` | Number of FL rounds (default: 100) |
| `batch_size` | Batch size (default: 32) |
| `alpha` | Dirichlet non-IID parameter (default: 0.1) |
| `seed` | Random seed (default: 1) |

### Tuning Other Hyperparameters

Navigate into the relevant subfolders:

1) Client's partial participation rate (fraction_fit): 

- fednova (CIFAR-10/100) > conf > strategy > fedavg.yaml/fednova.yaml 

2) Learning rate, mu, momentum, weight decay, gmf: 

- fednova (CIFAR-10/100) > conf > optimizer > fedavg_opt.yaml/fedprox_opt.yaml/fednova_prox_opt.yaml

### Running Locally

Navigate into the relevant subfolder (e.g. `fednova (CIFAR-10)/`), remove the parenthesis to run:

```bash
# Centralized baseline
PYTHONPATH=$(pwd) python centralized.py

# FedAvg
PYTHONPATH=$(pwd) python -m main num_rounds=100 strategy=fedavg optimizer=fedavg_opt

# FedProx
PYTHONPATH=$(pwd) python -m main num_rounds=100 strategy=fedavg optimizer=fedprox_opt

# FedNova
PYTHONPATH=$(pwd) python -m main num_rounds=100 strategy=fednova optimizer=fednova_prox_opt
```

### Running on HPC (SLURM)

> ⚠️ **Before running the `.sh` scripts**, update the following line in each script to your own HPC username and virtual environment path (the uploaded scripts were ran on NTU CCDS GPU Cluster TC2):
>
> ```bash
> # Change this (example — {PATH} is specific user's path with your actual HPC username):
> export PYTHONPATH=/{PATH}/fednova:$PYTHONPATH
> cd /{PATH}/fednova
> ```

Submit jobs with:

```bash
sbatch run_fedavg.sh
sbatch run_fednova.sh
sbatch run_fedprox.sh
sbatch run_centralized.sh
```

---

## Folder 2 — Privacy Attack Code 

Contains a Jupyter notebook that performs 4 privacy attacks on the trained FL models:

| Attack | Method |
|---|---|
| Attack A: PIA (Gradient-based) | Threshold + Meta-Classifier on gradient norm signals |
| Attack B: MIA (Confidence-based, Baseline) | Softmax confidence simulation vs Centralized baseline |
| Attack 3: Update Norm Inference | Correlation + Regression to infer client data size |
| Attack 4: Aggregate Participation Inference | Reconstruct which clients participated each round |

### How to Run

1. Open `final_privacy_attacks-3.ipynb` in Jupyter or VS Code
2. The required `.csv` and `.json` data files are already included in the folder (i.e. outputs from folder 1 & 3)
3. Run all cells **top to bottom**
4. If needed, edit the file paths in **Cell 5** (Section 1 of the notebook):

```python
PRIVACY_PATHS = {
    'FedAvg':  'fedavg_fedavg_varEpoch_True_seed_1_privacy.csv',
    'FedNova': 'fednova_prox_fednova_varEpoch_True_seed_1_privacy.csv',
    'FedProx': 'fedprox_fedavg_varEpoch_True_seed_1_privacy.csv',
}
PARTITION_PATH = 'partition_indices_seed1_alpha0.1_clients16.json'
```

> These paths assume the notebook and data files are in the **same folder**, which they are by default.

### Output

The notebook saves the following result files in the same directory:
- `results_gradient_attacks.csv`
- `results_confidence_mia.csv`
- `attack1a_mia_gradient.png`
- `attack1b_mia_confidence.png`
- `attack3_norm_inference.png`
- `attack4_participation.png`

> **Note:** Despite the renamed attacks (A/B instead of 1A/1B), the saved `.png` filenames remain `attack1a_mia_gradient.png` and `attack1b_mia_confidence.png`.

---

## Folder 3 — Privacy Defence Code 

Contains FL training with **Differential Privacy (DP-SGD)** via [Opacus](https://opacus.ai/), applied to the FedNova framework.

### Key Differences from Folder 1

- DP is enabled by default in `conf/base.yaml` under the `dp:` section
- Uses `SmallCNN` model (instead of VGG) for faster DP training
- Supports early stopping

### Configuration

DP settings in `conf/base.yaml`:

```yaml
dp:
  enabled: true          # Enable/disable DP-SGD
  noise_multiplier: 1.0  # Gaussian noise multiplier
  max_grad_norm: 1.5     # Per-sample gradient clipping bound
  target_delta: 1.0e-5   # Target delta for (ε, δ)-DP guarantee
```

### Running Locally

Navigate into `Privacy Defence Code/FEDNOVA/` and run:

```bash
# FedAvg with DP
PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fedavg optimizer=fedavg_opt

# FedProx with DP
PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fedavg optimizer=fedprox_opt

# FedNova with DP
PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fednova optimizer=fednova_prox_opt

# Centralized with DP
PYTHONPATH=$(pwd) python -m fednova.centralized
```

### Running on HPC (SLURM)

> ⚠️ **Before running the `.sh` scripts**, update the virtual environment activation line to your own path (the uploaded scripts were ran on NTU CCDS GPU Cluster TC2):
>
> ```bash
> # Change this (example — {PATH} is specific user's path with your actual HPC username):
> source /{PATH}/venvs/fednova310/bin/activate
> ```

Submit jobs with:

```bash
sbatch run_fedavg.sh
sbatch run_fednovaprox.sh
sbatch run_fedprox.sh
sbatch run_centralized.sh
```

---

## Notes

- All folders use **relative paths** — no changes needed when cloning to a new machine
- The `data/` folder is excluded from this repository (auto-downloaded on first run)
- Checkpoints, logs, and generated outputs are also excluded (see `.gitignore`)
- The `.sh` scripts are configured for our university HPC cluster (SLURM). Update the username/venv paths before use as described above
