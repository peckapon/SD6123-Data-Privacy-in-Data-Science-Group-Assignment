"""Entry script for Federated training on CIFAR-10 using FedNova/FedAvg/FedProx.

Supports checkpointing so training can resume across multiple GPU job submissions.
Final global model is saved for all strategies after training completes.

New outputs (privacy-attack support)
─────────────────────────────────────
*_privacy.csv  — per-round, per-client update norms + participation flags
*_grads.npz    — per-round aggregated update vectors (flat float32 arrays)

See strategy.py for the full output → attack mapping table.
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
    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    fname = (
        f"checkpoint_{cfg.exp_name}_{cfg.strategy.name}_"
        f"varEpoch_{cfg.var_local_epochs}_seed_{cfg.seed}.pkl"
    )
    return os.path.join(cfg.checkpoint_path, fname)


def _final_model_path(cfg: DictConfig) -> str:
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
    privacy_rows: list,
    grad_log: list,
    cumulative_bytes_sent: int,
    cumulative_bytes_received: int,
    cumulative_time_sec: float,
) -> None:
    state = {
        "completed_rounds": completed_rounds,
        "global_params": global_params,
        "results_rows": results_rows,
        "comm_rows": comm_rows,
        "timing_rows": timing_rows,
        "privacy_rows": privacy_rows,
        "grad_log": grad_log,
        "cumulative_bytes_sent": cumulative_bytes_sent,
        "cumulative_bytes_received": cumulative_bytes_received,
        "cumulative_time_sec": cumulative_time_sec,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved after round {completed_rounds} → {path}")


def _save_final_model(path: str, global_params: list, metrics: dict) -> None:
    np.savez(
        path,
        global_parameters=np.array(global_params, dtype=object),
        **{k: np.array(v) for k, v in metrics.items()},
    )
    print(f"[Final Model] Saved → {path}")


def _load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        state = pickle.load(f)
    print(f"[Checkpoint] Resuming from round {state['completed_rounds']} ← {path}")
    return state


# ── Custom strategy wrappers ───────────────────────────────────────────────────

class CheckpointedFedAvg(FedAvgWithCommTracking):
    """FedAvgWithCommTracking that checkpoints after every round.

    The key design: we replace the parent's fresh privacy_log and grad_log
    lists (created in FedAvgWithCommTracking.__init__) with the lists passed
    in from main(). Since aggregate_fit appends to self.privacy_log and
    self.grad_log, it writes directly into the main() lists with no sync needed.
    """

    def __init__(self, checkpoint_cfg, results_rows, comm_rows, timing_rows,
                 privacy_rows, grad_log, round_start_times, total_start,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ckpt_cfg = checkpoint_cfg
        self.results_rows = results_rows
        self.comm_rows = comm_rows
        self.timing_rows = timing_rows
        # Replace parent's fresh empty lists with ours from main()
        self.privacy_log = privacy_rows
        self.grad_log = grad_log
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
        if aggregated_parameters is not None:
            self._last_round_params = list(
                parameters_to_ndarrays(aggregated_parameters)
            )
        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)

        if result is not None and server_round > 0:
            loss, metrics = result

            elapsed = time.time() - self._round_start_times.get(
                server_round, self._total_start
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
                global_params=self._last_round_params,
                results_rows=self.results_rows,
                comm_rows=self.comm_rows,
                timing_rows=self.timing_rows,
                privacy_rows=self.privacy_log,
                grad_log=self.grad_log,
                cumulative_bytes_sent=self.cumulative_bytes_sent,
                cumulative_bytes_received=self.cumulative_bytes_received,
                cumulative_time_sec=cumulative,
            )

        return result


class CheckpointedFedNova(FedNova):
    """FedNova that checkpoints after every round.

    Same design as CheckpointedFedAvg: we replace FedNova's fresh privacy_log
    and grad_log lists (created in FedNova.__init__) with the lists from main()
    immediately after super().__init__ returns.
    """

    def __init__(self, checkpoint_cfg, results_rows, comm_rows, timing_rows,
                 privacy_rows, grad_log, round_start_times, total_start,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ckpt_cfg = checkpoint_cfg
        self.results_rows = results_rows
        self.comm_rows = comm_rows
        self.timing_rows = timing_rows
        # Replace parent's fresh empty lists with ours from main()
        self.privacy_log = privacy_rows
        self.grad_log = grad_log
        self._round_start_times = round_start_times
        self._total_start = total_start

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)

        if result is not None and server_round > 0:
            loss, metrics = result

            elapsed = time.time() - self._round_start_times.get(
                server_round, self._total_start
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
                global_params=list(self.global_parameters),
                results_rows=self.results_rows,
                comm_rows=self.comm_rows,
                timing_rows=self.timing_rows,
                privacy_rows=self.privacy_log,
                grad_log=self.grad_log,
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

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)

    print(OmegaConf.to_yaml(cfg))

    os.makedirs(cfg.datapath, exist_ok=True)
    os.makedirs(cfg.checkpoint_path, exist_ok=True)

    trainloaders, testloader, data_sizes = load_datasets(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Test-only mode ─────────────────────────────────────────────────────────
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

    # ── Resume / fresh start ───────────────────────────────────────────────────
    ckpt = _load_checkpoint(_checkpoint_path(cfg))
    completed_rounds = ckpt.get("completed_rounds", 0)
    remaining_rounds = cfg.num_rounds - completed_rounds

    if remaining_rounds <= 0:
        print(
            f"[Checkpoint] Training already complete "
            f"({completed_rounds}/{cfg.num_rounds} rounds)."
        )
        return None

    print(
        f"[Checkpoint] Starting from round {completed_rounds + 1} / {cfg.num_rounds} "
        f"({remaining_rounds} rounds remaining)"
    )

    results_rows: list = ckpt.get("results_rows", [])
    comm_rows: list    = ckpt.get("comm_rows", [])
    timing_rows: list  = ckpt.get("timing_rows", [])
    privacy_rows: list = ckpt.get("privacy_rows", [])
    grad_log: list     = ckpt.get("grad_log", [])

    if completed_rounds > 0:
        init_ndarrays = ckpt["global_params"]
        print(f"[Checkpoint] Restored global model from round {completed_rounds}")
    else:
        init_ndarrays = [
            layer_param.cpu().numpy()
            for _, layer_param in instantiate(cfg.model).state_dict().items()
        ]
    init_parameters = ndarrays_to_parameters(init_ndarrays)

    client_fn = call(
        cfg.strategy.client_fn,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        testloader=testloader,
        data_sizes=data_sizes,
        model=cfg.model,
        exp_config=cfg,
    )

    round_start_times: dict = {}

    def timed_fit_config(server_round: int) -> dict:
        global_round = completed_rounds + server_round
        round_start_times[global_round] = time.time()
        return fit_config(cfg, global_round)

    eval_fn = partial(test, instantiate(cfg.model), testloader, device)

    shared_kwargs = dict(
        checkpoint_cfg=cfg,
        results_rows=results_rows,
        comm_rows=comm_rows,
        timing_rows=timing_rows,
        privacy_rows=privacy_rows,
        grad_log=grad_log,
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

    if completed_rounds > 0:
        strategy.cumulative_bytes_sent = ckpt.get("cumulative_bytes_sent", 0)
        strategy.cumulative_bytes_received = ckpt.get("cumulative_bytes_received", 0)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=remaining_rounds),
        strategy=strategy,
        client_resources=cfg.client_resources,
        ray_init_args={"ignore_reinit_error": True},
    )

    total_elapsed = time.time() - total_start

    # ── Save all outputs ───────────────────────────────────────────────────────
    # privacy_rows and grad_log were passed by reference into the strategy,
    # so they are already fully populated — no sync needed.
    save_path = HydraConfig.get().runtime.output_dir
    exp_tag = (
        f"{cfg.exp_name}_{cfg.strategy.name}_varEpoch_"
        f"{cfg.var_local_epochs}_seed_{cfg.seed}"
    )

    results_df = pd.DataFrame(results_rows).sort_values("round").reset_index(drop=True)
    comm_df    = pd.DataFrame(comm_rows).sort_values("round").reset_index(drop=True)

    privacy_df = pd.DataFrame(privacy_rows)
    if not privacy_df.empty:
        privacy_df = (
            privacy_df.sort_values(["round", "client_id"])
            .reset_index(drop=True)
        )

    results_file = os.path.join(save_path, f"{exp_tag}_results.csv")
    comm_file    = os.path.join(save_path, f"{exp_tag}_comm.csv")
    privacy_file = os.path.join(save_path, f"{exp_tag}_privacy.csv")
    grads_file   = os.path.join(save_path, f"{exp_tag}_grads.npz")

    results_df.to_csv(results_file, index=False)
    comm_df.to_csv(comm_file, index=False)
    privacy_df.to_csv(privacy_file, index=False)

    if grad_log:
        np.savez(grads_file, **{
            f"round_{entry['round']}": entry["agg_update"]
            for entry in grad_log
        })
        print(f"Gradient vectors saved  → {grads_file}")

    print(f"Results saved           → {results_file}")
    print(f"Communication log       → {comm_file}")
    print(f"Privacy / norm log      → {privacy_file}")
    print(results_df.tail())

    if results_df.empty:
        print("[Warning] No results recorded — skipping final model save.")
        return None

    final_params = (
        strategy._last_round_params
        if hasattr(strategy, "_last_round_params")
        else strategy.global_parameters
    )
    final_metrics = {
        "test_loss":      results_df["test_loss"].iloc[-1],
        "test_accuracy":  results_df["test_accuracy"].iloc[-1],
        "test_f1":        results_df["test_f1"].iloc[-1],
        "test_precision": results_df["test_precision"].iloc[-1],
        "test_recall":    results_df["test_recall"].iloc[-1],
    }
    _save_final_model(_final_model_path(cfg), final_params, final_metrics)

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
