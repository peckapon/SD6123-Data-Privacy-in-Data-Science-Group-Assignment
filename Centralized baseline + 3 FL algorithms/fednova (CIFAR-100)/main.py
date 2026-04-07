"""Entry script for Federated training on CIFAR-10 using FedNova/FedAvg/FedProx.

Supports checkpointing so training can resume across multiple GPU job submissions.
Final global model is saved for all strategies after training completes.
"""

import os
import pickle
import random
import time
from collections import OrderedDict
from functools import partial

import flwr as fl
import hydra
import numpy as np
import pandas as pd
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from fednova.dataset import load_datasets
from fednova.models import test
from fednova.strategy import FedAvgWithCommTracking, FedNova, weighted_average
from fednova.utils import fit_config


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _checkpoint_path(cfg: DictConfig) -> str:
    """Return the path to the resume checkpoint file for this experiment."""
    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    fname = (
        f"checkpoint_{cfg.exp_name}_{cfg.strategy.name}_"
        f"varEpoch_{cfg.var_local_epochs}_seed_{cfg.seed}.pkl"
    )
    return os.path.join(cfg.checkpoint_path, fname)


def _final_model_path(cfg: DictConfig) -> str:
    """Return the path to the final model .npz file for this experiment."""
    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    fname = (
        f"finalModel_{cfg.exp_name}_{cfg.strategy.name}_"
        f"varEpoch_{cfg.var_local_epochs}_seed_{cfg.seed}.npz"
    )
    return os.path.join(cfg.checkpoint_path, fname)


def _save_checkpoint(
    path: str,
    completed_rounds: int,
    global_params: list,
    results_rows: list,
    comm_rows: list,
    timing_rows: list,
    cumulative_bytes_sent: int,
    cumulative_bytes_received: int,
    cumulative_time_sec: float,
) -> None:
    """Save all state needed to resume training."""
    state = {
        "completed_rounds": completed_rounds,
        "global_params": global_params,
        "results_rows": results_rows,
        "comm_rows": comm_rows,
        "timing_rows": timing_rows,
        "cumulative_bytes_sent": cumulative_bytes_sent,
        "cumulative_bytes_received": cumulative_bytes_received,
        "cumulative_time_sec": cumulative_time_sec,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    # Atomic replace so a crash mid-save doesn't corrupt the checkpoint
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved after round {completed_rounds} → {path}")


def _save_final_model(path: str, global_params: list, metrics: dict) -> None:
    """Save the final global model parameters to a .npz file."""
    np.savez(
        path,
        global_parameters=np.array(global_params, dtype=object),
        **{k: np.array(v) for k, v in metrics.items()},
    )
    print(f"[Final Model] Saved → {path}")


def _load_checkpoint(path: str) -> dict:
    """Load checkpoint state. Returns empty dict if no checkpoint exists."""
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        state = pickle.load(f)
    print(
        f"[Checkpoint] Resuming from round {state['completed_rounds']} ← {path}"
    )
    return state


# ── Custom strategy wrappers that checkpoint after every round ─────────────────

class CheckpointedFedAvg(FedAvgWithCommTracking):
    """FedAvgWithCommTracking that saves a checkpoint after every round."""

    def __init__(self, checkpoint_cfg, results_rows, comm_rows, timing_rows,
                 round_start_times, total_start, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ckpt_cfg = checkpoint_cfg
        self.results_rows = results_rows
        self.comm_rows = comm_rows
        self.timing_rows = timing_rows
        self._round_start_times = round_start_times
        self._total_start = total_start
        self._last_round_params = (
            list(parameters_to_ndarrays(self.initial_parameters))
            if self.initial_parameters else []
        )

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        # Keep a copy of the latest aggregated global parameters
        if aggregated_parameters is not None:
            self._last_round_params = list(
                parameters_to_ndarrays(aggregated_parameters)
            )
        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)

        if result is not None and server_round > 0:
            loss, metrics = result

            # Timing
            elapsed = (
                time.time() - self._round_start_times.get(
                    server_round, self._total_start
                )
            )
            cumulative = (
                (self.timing_rows[-1]["cumulative_time_sec"] if self.timing_rows else 0)
                + elapsed
            )
            self.timing_rows.append({
                "round": server_round,
                "round_time_sec": round(elapsed, 2),
                "cumulative_time_sec": round(cumulative, 2),
            })

            # Results
            self.results_rows.append({
                "round": server_round,
                "test_loss": loss,
                "test_accuracy": metrics.get("accuracy", float("nan")),
                "test_f1": metrics.get("f1", float("nan")),
                "test_precision": metrics.get("precision", float("nan")),
                "test_recall": metrics.get("recall", float("nan")),
                "round_time_sec": round(elapsed, 2),
                "cumulative_time_sec": round(cumulative, 2),
            })

            # Comm (latest comm_log entry)
            if self.comm_log:
                self.comm_rows.append(self.comm_log[-1])

            # Save resume checkpoint
            _save_checkpoint(
                _checkpoint_path(self._ckpt_cfg),
                completed_rounds=server_round,
                global_params=self._last_round_params,
                results_rows=self.results_rows,
                comm_rows=self.comm_rows,
                timing_rows=self.timing_rows,
                cumulative_bytes_sent=self.cumulative_bytes_sent,
                cumulative_bytes_received=self.cumulative_bytes_received,
                cumulative_time_sec=cumulative,
            )

        return result


class CheckpointedFedNova(FedNova):
    """FedNova that saves a checkpoint after every round."""

    def __init__(self, checkpoint_cfg, results_rows, comm_rows, timing_rows,
                 round_start_times, total_start, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ckpt_cfg = checkpoint_cfg
        self.results_rows = results_rows
        self.comm_rows = comm_rows
        self.timing_rows = timing_rows
        self._round_start_times = round_start_times
        self._total_start = total_start

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)

        if result is not None and server_round > 0:
            loss, metrics = result

            elapsed = (
                time.time() - self._round_start_times.get(
                    server_round, self._total_start
                )
            )
            cumulative = (
                (self.timing_rows[-1]["cumulative_time_sec"] if self.timing_rows else 0)
                + elapsed
            )
            self.timing_rows.append({
                "round": server_round,
                "round_time_sec": round(elapsed, 2),
                "cumulative_time_sec": round(cumulative, 2),
            })

            self.results_rows.append({
                "round": server_round,
                "test_loss": loss,
                "test_accuracy": metrics.get("accuracy", float("nan")),
                "test_f1": metrics.get("f1", float("nan")),
                "test_precision": metrics.get("precision", float("nan")),
                "test_recall": metrics.get("recall", float("nan")),
                "round_time_sec": round(elapsed, 2),
                "cumulative_time_sec": round(cumulative, 2),
            })

            if self.comm_log:
                self.comm_rows.append(self.comm_log[-1])

            _save_checkpoint(
                _checkpoint_path(self._ckpt_cfg),
                completed_rounds=server_round,
                global_params=self.global_parameters,
                results_rows=self.results_rows,
                comm_rows=self.comm_rows,
                timing_rows=self.timing_rows,
                cumulative_bytes_sent=self.cumulative_bytes_sent,
                cumulative_bytes_received=self.cumulative_bytes_received,
                cumulative_time_sec=cumulative,
            )

        return result


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    """Run the baseline, resuming from checkpoint if one exists."""
    total_start = time.time()

    # 1. Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare directories
    if not os.path.exists(cfg.datapath):
        os.makedirs(cfg.datapath)
    if not os.path.exists(cfg.checkpoint_path):
        os.makedirs(cfg.checkpoint_path)

    trainloaders, testloader, data_sizes = load_datasets(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Test-only mode ─────────────────────────────────────────────────────
    if cfg.mode == "test":
        checkpoint = np.load(
            f"{cfg.checkpoint_path}bestModel_{cfg.exp_name}_"
            f"{cfg.strategy.name}_varEpochs_{cfg.var_local_epochs}.npz",
            allow_pickle=True,
        )
        model = instantiate(cfg.model)
        params_dict = zip(model.state_dict().keys(), checkpoint["global_parameters"])
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict)
        loss, metrics = test(model.to(device), testloader, device)
        print(
            f"---- Loss: {loss:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} ----"
        )
        return None

    # 3. Load checkpoint if available
    ckpt = _load_checkpoint(_checkpoint_path(cfg))
    completed_rounds = ckpt.get("completed_rounds", 0)
    remaining_rounds = cfg.num_rounds - completed_rounds

    if remaining_rounds <= 0:
        print(
            f"[Checkpoint] Training already complete "
            f"({completed_rounds}/{cfg.num_rounds} rounds). Nothing to do."
        )
        return None

    print(
        f"[Checkpoint] Starting from round {completed_rounds + 1} / {cfg.num_rounds} "
        f"({remaining_rounds} rounds remaining)"
    )

    # Restore accumulated logs from checkpoint (empty lists if fresh start)
    results_rows: list = ckpt.get("results_rows", [])
    comm_rows: list = ckpt.get("comm_rows", [])
    timing_rows: list = ckpt.get("timing_rows", [])

    # 4. Initial parameters — use checkpoint params if resuming, else random init
    if completed_rounds > 0:
        init_ndarrays = ckpt["global_params"]
        print(f"[Checkpoint] Restored global model from round {completed_rounds}")
    else:
        init_ndarrays = [
            layer_param.cpu().numpy()
            for _, layer_param in instantiate(cfg.model).state_dict().items()
        ]
    init_parameters = ndarrays_to_parameters(init_ndarrays)

    # 5. Build client function
    client_fn = call(
        cfg.strategy.client_fn,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        testloader=testloader,
        data_sizes=data_sizes,
        model=cfg.model,
        exp_config=cfg,
    )

    # 6. Per-round timing
    round_start_times: dict = {}

    def timed_fit_config(server_round: int) -> dict:
        """Record round start time and return fit config."""
        # Map relative server_round back to global round number
        global_round = completed_rounds + server_round
        round_start_times[global_round] = time.time()
        return fit_config(cfg, global_round)

    # 7. Evaluation function
    eval_fn = partial(test, instantiate(cfg.model), testloader, device)

    # 8. Build strategy with checkpointing
    shared_kwargs = dict(
        checkpoint_cfg=cfg,
        results_rows=results_rows,
        comm_rows=comm_rows,
        timing_rows=timing_rows,
        round_start_times=round_start_times,
        total_start=total_start,
        evaluate_metrics_aggregation_fn=weighted_average,
        accept_failures=False,
        on_fit_config_fn=timed_fit_config,
        initial_parameters=init_parameters,
        evaluate_fn=eval_fn,
        fraction_evaluate=0.0,
        fraction_fit=cfg.strategy.strategy.fraction_fit,
    )

    if cfg.strategy.name == "fedavg":
        strategy = CheckpointedFedAvg(**shared_kwargs)
    else:
        strategy = CheckpointedFedNova(exp_config=cfg, **shared_kwargs)

    # Restore cumulative comm bytes if resuming
    if completed_rounds > 0:
        strategy.cumulative_bytes_sent = ckpt.get("cumulative_bytes_sent", 0)
        strategy.cumulative_bytes_received = ckpt.get("cumulative_bytes_received", 0)

    # 9. Run simulation for remaining rounds only
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=remaining_rounds),
        strategy=strategy,
        client_resources=cfg.client_resources,
        ray_init_args={"ignore_reinit_error": True},
    )

    total_elapsed = time.time() - total_start

    # 10. Save final CSVs (full history across all jobs)
    save_path = HydraConfig.get().runtime.output_dir

    results_df = pd.DataFrame(results_rows).sort_values("round").reset_index(drop=True)
    comm_df = pd.DataFrame(comm_rows).sort_values("round").reset_index(drop=True)

    results_file = os.path.join(
        save_path,
        f"{cfg.exp_name}_{cfg.strategy.name}_varEpoch_"
        f"{cfg.var_local_epochs}_seed_{cfg.seed}_results.csv",
    )
    comm_file = os.path.join(
        save_path,
        f"{cfg.exp_name}_{cfg.strategy.name}_varEpoch_"
        f"{cfg.var_local_epochs}_seed_{cfg.seed}_comm.csv",
    )

    results_df.to_csv(results_file, index=False)
    comm_df.to_csv(comm_file, index=False)

    print(f"\nResults saved → {results_file}")
    print(f"Communication log saved → {comm_file}")
    print(results_df.tail())

    # 11. Save final global model for all strategies
    #     FedNova already saves best model in strategy.py — here we additionally
    #     save the FINAL round model for all strategies (needed for MIA later)
    if results_df.empty:
        print("[Warning] No results recorded — skipping final model save.")
        return None

    final_params = (
        strategy._last_round_params          # FedAvg / FedProx
        if hasattr(strategy, "_last_round_params")
        else strategy.global_parameters      # FedNova
    )
    final_metrics = {
        "test_loss": results_df["test_loss"].iloc[-1],
        "test_accuracy": results_df["test_accuracy"].iloc[-1],
        "test_f1": results_df["test_f1"].iloc[-1],
        "test_precision": results_df["test_precision"].iloc[-1],
        "test_recall": results_df["test_recall"].iloc[-1],
    }
    _save_final_model(_final_model_path(cfg), final_params, final_metrics)

    # 12. Final summary
    print(
        f"\n{'='*60}\n"
        f"Experiment        : {cfg.exp_name} / {cfg.strategy.name}\n"
        f"Rounds this job   : {remaining_rounds}\n"
        f"Total rounds done : {completed_rounds + remaining_rounds}\n"
        f"Time this job     : {total_elapsed / 60:.2f} minutes\n"
        f"Final accuracy    : {results_df['test_accuracy'].iloc[-1]:.4f}\n"
        f"Final F1          : {results_df['test_f1'].iloc[-1]:.4f}\n"
        f"Final precision   : {results_df['test_precision'].iloc[-1]:.4f}\n"
        f"Final recall      : {results_df['test_recall'].iloc[-1]:.4f}\n"
        f"Bytes sent        : {strategy.cumulative_bytes_sent / 1e6:.2f} MB\n"
        f"Bytes received    : {strategy.cumulative_bytes_received / 1e6:.2f} MB\n"
        f"{'='*60}"
    )

    return None


if __name__ == "__main__":
    main()
