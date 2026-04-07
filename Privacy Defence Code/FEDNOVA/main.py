"""
Entry script for Federated training on CIFAR-10 using FedNova/FedAvg/FedProx.

- Supports checkpointing so training can resume across multiple GPU job submissions.
- Final global model is saved for all strategies after training completes.

New outputs (privacy-attack support):
─────────────────────────────────────
*_privacy.csv  — per-round, per-client update norms + participation flags
*_grads.npz    — per-round aggregated update vectors (flat float32 arrays)

See strategy.py for the full output → attack mapping table.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

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
    """
    Build the checkpoint file path for saving training state.
    """
    os.makedirs(cfg.checkpoint_path, exist_ok=True)
    fname = (
        f"checkpoint_{cfg.exp_name}_{cfg.strategy.name}_"
        f"varEpoch_{cfg.var_local_epochs}_seed_{cfg.seed}.pkl"
    )
    return os.path.join(cfg.checkpoint_path, fname)


def _final_model_path(cfg: DictConfig) -> str:
    """
    Build the file path for saving the final model.
    """
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
    epsilon_rows: list | None = None,
) -> None:
    """
    Save training checkpoint to disk, including all logs and model state.
    """
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
        "epsilon_rows": epsilon_rows or [],
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved after round {completed_rounds} → {path}")


def _save_final_model(path: str, global_params: list, metrics: dict) -> None:
    """
    Save the final model parameters and metrics to a .npz file.
    """
    np.savez(
        path,
        global_parameters=np.array(global_params, dtype=object),
        **{k: np.array(v) for k, v in metrics.items()},
    )
    print(f"[Final Model] Saved → {path}")


def _load_checkpoint(path: str) -> dict:
    """
    Load checkpoint from disk if it exists, otherwise return empty dict.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        state = pickle.load(f)
    print(f"[Checkpoint] Resuming from round {state['completed_rounds']} ← {path}")
    return state


def _replay_epsilon_history(accountant, epsilon_rows: list[dict]) -> None:
    """
    Restore composed DP history into a fresh server-side accountant.
    """
    for entry in epsilon_rows:
        dp_steps = int(entry.get("dp_num_steps", 0))
        if dp_steps <= 0:
            continue
        accountant.history.append(
            (
                float(entry.get("dp_noise_multiplier", 0.0)),
                float(entry.get("dp_sample_rate", 0.0)),
                dp_steps,
            )
        )


def _get_early_stop_state(results_rows: list[dict], cfg: DictConfig) -> dict:
    """
    Return current early-stop status based on centralized test loss.
    """
    default_state = {
        "triggered": False,
        "best_loss": float("inf"),
        "best_round": 0,
        "bad_rounds": 0,
        "stop_round": None,
        "reason": "",
    }
    if not getattr(cfg.early_stop, "enabled", False) or not results_rows:
        return default_state

    rows = sorted(results_rows, key=lambda row: row["round"])
    patience = int(getattr(cfg.early_stop, "patience", 0))
    warmup_rounds = int(getattr(cfg.early_stop, "warmup_rounds", 0))
    min_delta = float(getattr(cfg.early_stop, "min_delta", 0.0))

    best_loss = float("inf")
    best_round = 0
    bad_rounds = 0

    for row in rows:
        round_id = int(row["round"])
        loss = float(row["test_loss"])
        improved = loss < (best_loss - min_delta)
        if improved:
            best_loss = loss
            best_round = round_id
            bad_rounds = 0
        elif round_id > warmup_rounds:
            bad_rounds += 1

        if round_id > warmup_rounds and bad_rounds >= patience:
            return {
                "triggered": True,
                "best_loss": best_loss,
                "best_round": best_round,
                "bad_rounds": bad_rounds,
                "stop_round": round_id,
                "reason": (
                    "No significant test-loss improvement for "
                    f"{patience} round(s) after round {best_round}."
                ),
            }

    return {
        "triggered": False,
        "best_loss": best_loss,
        "best_round": best_round,
        "bad_rounds": bad_rounds,
        "stop_round": None,
        "reason": "",
    }


# ── Custom strategy wrappers ───────────────────────────────────────────────────

class CheckpointedFedAvg(FedAvgWithCommTracking):
    """FedAvgWithCommTracking that checkpoints after every round.

    The key design: we replace the parent's fresh privacy_log, grad_log, and
    epsilon_log lists (created in FedAvgWithCommTracking.__init__) with the
    lists passed in from main(). Since aggregate_fit appends to these,
    they write directly into the main() lists with no sync needed.
    """

    def __init__(self, checkpoint_cfg, results_rows, comm_rows, timing_rows,
                 privacy_rows, grad_log, epsilon_rows, round_start_times, total_start,
                 round_offset,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ckpt_cfg = checkpoint_cfg
        self.results_rows = results_rows
        self.comm_rows = comm_rows
        self.timing_rows = timing_rows
        # Replace parent's fresh empty lists with ours from main()
        self.privacy_log = privacy_rows
        self.grad_log = grad_log
        self.epsilon_log = epsilon_rows
        self._round_start_times = round_start_times
        self._total_start = total_start
        self._round_offset = round_offset
        self._last_round_params = (
            list(parameters_to_ndarrays(self.initial_parameters))
            if self.initial_parameters else []
        )
        # Server-side RDP accountant for cumulative ε tracking
        if checkpoint_cfg.dp.enabled:
            from opacus.accountants import RDPAccountant  # pylint: disable=import-outside-toplevel
            self._rdp_accountant = RDPAccountant()
            _replay_epsilon_history(self._rdp_accountant, epsilon_rows)
        else:
            self._rdp_accountant = None
        self._early_stop_triggered = False
        self._early_stop_message = ""

    def configure_fit(self, server_round, parameters, client_manager):
        # Once early stop is triggered, skip remaining rounds without crashing
        # Flower; the server will quickly finish the configured total rounds.
        if self._early_stop_triggered:
            return []
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        global_round = self._round_offset + server_round
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Strategy internals log local rounds; remap newest entries to global round.
        if self.comm_log and self.comm_log[-1].get("round") == server_round:
            self.comm_log[-1]["round"] = global_round
        if self.grad_log and self.grad_log[-1].get("round") == server_round:
            self.grad_log[-1]["round"] = global_round
        if self.epsilon_log and self.epsilon_log[-1].get("round") == server_round:
            self.epsilon_log[-1]["round"] = global_round
        for row in self.privacy_log[-len(results):]:
            if row.get("round") == server_round:
                row["round"] = global_round

        if aggregated_parameters is not None:
            self._last_round_params = list(
                parameters_to_ndarrays(aggregated_parameters)
            )
        # Compute cumulative ε via RDP composition after aggregate_fit updates epsilon_log
        if self._rdp_accountant is not None and self.epsilon_log:
            entry = self.epsilon_log[-1]
            if entry["round"] == global_round:
                self._rdp_accountant.history.append((
                    entry["dp_noise_multiplier"],
                    entry["dp_sample_rate"],
                    entry["dp_num_steps"],
                ))
                cumulative_eps = self._rdp_accountant.get_epsilon(
                    delta=self._ckpt_cfg.dp.target_delta
                )
                entry["cumulative_epsilon"] = float(cumulative_eps)
                print(
                    f"[DP] Round {global_round}: "
                    f"\u03b5_round={entry['mean_epsilon_this_round']:.4f}  "
                    f"\u03b5_cumulative={cumulative_eps:.4f}  "
                    f"(\u03b4={self._ckpt_cfg.dp.target_delta:.0e})"
                )
        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round, parameters):
        if self._early_stop_triggered:
            return None

        global_round = self._round_offset + server_round
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
                "round": global_round,
                "round_time_sec": round(elapsed, 2),
                "cumulative_time_sec": round(cumulative, 2),
            })
            self.results_rows.append({
                "round": global_round,
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
                completed_rounds=global_round,
                global_params=self._last_round_params,
                results_rows=self.results_rows,
                comm_rows=self.comm_rows,
                timing_rows=self.timing_rows,
                privacy_rows=self.privacy_log,
                grad_log=self.grad_log,
                cumulative_bytes_sent=self.cumulative_bytes_sent,
                cumulative_bytes_received=self.cumulative_bytes_received,
                cumulative_time_sec=cumulative,
                epsilon_rows=self.epsilon_log,
            )

            if getattr(self._ckpt_cfg.early_stop, "enabled", False):
                es = _get_early_stop_state(self.results_rows, self._ckpt_cfg)
                if es["triggered"] and not self._early_stop_triggered:
                    self._early_stop_triggered = True
                    self._early_stop_message = (
                        f"[Early Stop] Triggered at round {es['stop_round']}: "
                        f"{es['reason']}"
                    )
                    print(self._early_stop_message)

        return result


class CheckpointedFedNova(FedNova):
    """FedNova that checkpoints after every round.

    Same design as CheckpointedFedAvg: we replace FedNova's fresh privacy_log,
    grad_log, and epsilon_log lists (created in FedNova.__init__) with the
    lists from main() immediately after super().__init__ returns.
    """

    def __init__(self, checkpoint_cfg, results_rows, comm_rows, timing_rows,
                 privacy_rows, grad_log, epsilon_rows, round_start_times, total_start,
                 round_offset,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ckpt_cfg = checkpoint_cfg
        self.results_rows = results_rows
        self.comm_rows = comm_rows
        self.timing_rows = timing_rows
        # Replace parent's fresh empty lists with ours from main()
        self.privacy_log = privacy_rows
        self.grad_log = grad_log
        self.epsilon_log = epsilon_rows
        self._round_start_times = round_start_times
        self._total_start = total_start
        self._round_offset = round_offset
        # Server-side RDP accountant for cumulative ε tracking
        if checkpoint_cfg.dp.enabled:
            from opacus.accountants import RDPAccountant  # pylint: disable=import-outside-toplevel
            self._rdp_accountant = RDPAccountant()
            _replay_epsilon_history(self._rdp_accountant, epsilon_rows)
        else:
            self._rdp_accountant = None
        self._early_stop_triggered = False
        self._early_stop_message = ""

    def configure_fit(self, server_round, parameters, client_manager):
        # Once early stop is triggered, skip remaining rounds without crashing
        # Flower; the server will quickly finish the configured total rounds.
        if self._early_stop_triggered:
            return []
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        global_round = self._round_offset + server_round
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Strategy internals log local rounds; remap newest entries to global round.
        if self.comm_log and self.comm_log[-1].get("round") == server_round:
            self.comm_log[-1]["round"] = global_round
        if self.grad_log and self.grad_log[-1].get("round") == server_round:
            self.grad_log[-1]["round"] = global_round
        if self.epsilon_log and self.epsilon_log[-1].get("round") == server_round:
            self.epsilon_log[-1]["round"] = global_round
        for row in self.privacy_log[-len(results):]:
            if row.get("round") == server_round:
                row["round"] = global_round

        # Compute cumulative ε via RDP composition after aggregate_fit updates epsilon_log
        if self._rdp_accountant is not None and self.epsilon_log:
            entry = self.epsilon_log[-1]
            if entry["round"] == global_round:
                self._rdp_accountant.history.append((
                    entry["dp_noise_multiplier"],
                    entry["dp_sample_rate"],
                    entry["dp_num_steps"],
                ))
                cumulative_eps = self._rdp_accountant.get_epsilon(
                    delta=self._ckpt_cfg.dp.target_delta
                )
                entry["cumulative_epsilon"] = float(cumulative_eps)
                print(
                    f"[DP] Round {global_round}: "
                    f"\u03b5_round={entry['mean_epsilon_this_round']:.4f}  "
                    f"\u03b5_cumulative={cumulative_eps:.4f}  "
                    f"(\u03b4={self._ckpt_cfg.dp.target_delta:.0e})"
                )
        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round, parameters):
        if self._early_stop_triggered:
            return None

        global_round = self._round_offset + server_round
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
                "round": global_round,
                "round_time_sec": round(elapsed, 2),
                "cumulative_time_sec": round(cumulative, 2),
            })
            self.results_rows.append({
                "round": global_round,
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
                completed_rounds=global_round,
                global_params=list(self.global_parameters),
                results_rows=self.results_rows,
                comm_rows=self.comm_rows,
                timing_rows=self.timing_rows,
                privacy_rows=self.privacy_log,
                grad_log=self.grad_log,
                cumulative_bytes_sent=self.cumulative_bytes_sent,
                cumulative_bytes_received=self.cumulative_bytes_received,
                cumulative_time_sec=cumulative,
                epsilon_rows=self.epsilon_log,
            )

            if getattr(self._ckpt_cfg.early_stop, "enabled", False):
                es = _get_early_stop_state(self.results_rows, self._ckpt_cfg)
                if es["triggered"] and not self._early_stop_triggered:
                    self._early_stop_triggered = True
                    self._early_stop_message = (
                        f"[Early Stop] Triggered at round {es['stop_round']}: "
                        f"{es['reason']}"
                    )
                    print(self._early_stop_message)

        return result


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    """Run the baseline, resuming from checkpoint if one exists."""
    total_start = time.time()
    initial_completed_rounds = 0

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
    initial_completed_rounds = completed_rounds
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

    results_rows: list  = ckpt.get("results_rows", [])
    comm_rows: list     = ckpt.get("comm_rows", [])
    timing_rows: list   = ckpt.get("timing_rows", [])
    privacy_rows: list  = ckpt.get("privacy_rows", [])
    grad_log: list      = ckpt.get("grad_log", [])
    epsilon_rows: list  = ckpt.get("epsilon_rows", [])

    early_stop_state = _get_early_stop_state(results_rows, cfg)
    if early_stop_state["triggered"]:
        print(
            f"[Early Stop] Already satisfied at round {early_stop_state['stop_round']}. "
            f"{early_stop_state['reason']}"
        )
        return None

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

    cumulative_bytes_sent = ckpt.get("cumulative_bytes_sent", 0)
    cumulative_bytes_received = ckpt.get("cumulative_bytes_received", 0)
    current_parameters = init_parameters
    strategy = None

    def _build_strategy(round_offset: int, init_params):
        shared_kwargs = dict(
            checkpoint_cfg=cfg,
            results_rows=results_rows,
            comm_rows=comm_rows,
            timing_rows=timing_rows,
            privacy_rows=privacy_rows,
            grad_log=grad_log,
            epsilon_rows=epsilon_rows,
            round_start_times=round_start_times,
            total_start=total_start,
            round_offset=round_offset,
            evaluate_metrics_aggregation_fn=weighted_average,
            accept_failures=False,
            on_fit_config_fn=timed_fit_config,
            initial_parameters=init_params,
            evaluate_fn=eval_fn,
            fraction_evaluate=0.0,
            fraction_fit=cfg.strategy.strategy.fraction_fit,
        )

        if cfg.strategy.name == "fedavg":
            strategy_local = CheckpointedFedAvg(**shared_kwargs)
        else:
            strategy_local = CheckpointedFedNova(exp_config=cfg, **shared_kwargs)

        strategy_local.cumulative_bytes_sent = cumulative_bytes_sent
        strategy_local.cumulative_bytes_received = cumulative_bytes_received
        return strategy_local

    strategy = _build_strategy(completed_rounds, current_parameters)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=remaining_rounds),
        strategy=strategy,
        client_resources=cfg.client_resources,
        ray_init_args={"ignore_reinit_error": True},
    )

    cumulative_bytes_sent = strategy.cumulative_bytes_sent
    cumulative_bytes_received = strategy.cumulative_bytes_received
    current_parameters = ndarrays_to_parameters(
        strategy._last_round_params
        if hasattr(strategy, "_last_round_params")
        else strategy.global_parameters
    )
    completed_rounds = int(results_rows[-1]["round"]) if results_rows else completed_rounds

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
    epsilon_file = os.path.join(save_path, f"{exp_tag}_dp_epsilon.csv")

    results_df.to_csv(results_file, index=False)
    comm_df.to_csv(comm_file, index=False)
    privacy_df.to_csv(privacy_file, index=False)

    epsilon_df = pd.DataFrame(epsilon_rows)
    if not epsilon_df.empty:
        epsilon_df = epsilon_df.sort_values("round").reset_index(drop=True)
        epsilon_df.to_csv(epsilon_file, index=False)
        print(f"DP epsilon log saved    \u2192 {epsilon_file}")

    if grad_log:
        np.savez(grads_file, **{
            f"round_{entry['round']}": entry["agg_update"]
            for entry in grad_log
        })
        print(f"Gradient vectors saved  → {grads_file}")

    print(f"Results saved           \u2192 {results_file}")
    print(f"Communication log       \u2192 {comm_file}")
    print(f"Privacy / norm log      \u2192 {privacy_file}")
    print(results_df.tail())

    if results_df.empty:
        print("[Warning] No results recorded — skipping final model save.")
        return None

    final_params = parameters_to_ndarrays(current_parameters)
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
        f"Rounds this job   : {completed_rounds - initial_completed_rounds}\n"
        f"Total rounds done : {completed_rounds}\n"
        f"Time this job     : {total_elapsed / 60:.2f} minutes\n"
        f"Final accuracy    : {results_df['test_accuracy'].iloc[-1]:.4f}\n"
        f"Final F1          : {results_df['test_f1'].iloc[-1]:.4f}\n"
        f"Final precision   : {results_df['test_precision'].iloc[-1]:.4f}\n"
        f"Final recall      : {results_df['test_recall'].iloc[-1]:.4f}\n"
        f"Bytes sent        : {cumulative_bytes_sent / 1e6:.2f} MB\n"
        f"Bytes received    : {cumulative_bytes_received / 1e6:.2f} MB\n"
        f"{'='*60}"
    )

    return None


if __name__ == "__main__":
    main()
