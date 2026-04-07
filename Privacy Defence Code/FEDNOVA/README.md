# FEDNOVA: Federated and Centralized Training Framework

## Getting Started

This repository supports both federated and centralized training for CIFAR-10, with flexible configuration for strategies, optimizers, and differential privacy (DP).

### 1. Environment Setup
- Activate your Python environment (see the `.sh` scripts for example paths):
  ```bash
  source /path/to/your/venv/bin/activate
  ```
- Ensure all dependencies are installed (see requirements in your project).

### 2. Configuration
- All main experiment settings are controlled in [`conf/base.yaml`](conf/base.yaml):
  - **Training parameters:** `num_rounds`, `num_epochs`, `batch_size`, `seed`, etc.
  - **Differential Privacy:** Enable/disable DP, set noise multiplier, grad norm, etc. under the `dp:` section.
  - **Model selection:** Choose the model class under the `model:` section.
  - **Strategy & Optimizer:**
    - The federated learning strategy is set under `strategy:` (e.g., `fedavg`, `fednova`).
    - The optimizer is set under `optimizer:` (e.g., `fedavg_opt`, `fedprox_opt`, `fednova_prox_opt`).

### 3. Running Training

#### Federated Learning
- **FedAvg:**
  ```bash
  PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fedavg optimizer=fedavg_opt
  ```
- **FedProx:**
  ```bash
  PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fedavg optimizer=fedprox_opt
  ```
- **FedNova (with Proximal):**
  ```bash
  PYTHONPATH=$(pwd) python -m fednova.main num_rounds=100 strategy=fednova optimizer=fednova_prox_opt
  ```

#### Centralized Training
- **Centralized:**
  ```bash
  PYTHONPATH=$(pwd) python -m fednova.centralized
  ```
  - You can add CLI arguments to override config values, e.g. `--model smallcnn --num_rounds 100`.

### 4. Customization
- **All configuration options** (including DP, model, strategy, optimizer, and training parameters) can be toggled in [`conf/base.yaml`](conf/base.yaml).
- For advanced users, you can override any config value via the command line (Hydra/OmegaConf style), e.g.:
  ```bash
  PYTHONPATH=$(pwd) python -m fednova.main num_rounds=200 dp.enabled=false strategy=fedavg optimizer=fedprox_opt
  ```

### 5. Output
- Results, checkpoints, and logs are saved to the paths specified in the config (`checkpoint_path`, `output_path`).

---

## Example SLURM Scripts
- See `run_fedavg.sh`, `run_fedprox.sh`, `run_fednovaprox.sh`, and `run_centralized.sh` for job submission examples on a cluster.

---

## Notes
- All main toggles and experiment settings are in [`conf/base.yaml`](conf/base.yaml).
- Strategies and optimizers can be switched by changing the `strategy:` and `optimizer:` fields in the config or via CLI.
- Differential privacy is supported for both federated and centralized training.
- For more details, see code comments and configuration files.
