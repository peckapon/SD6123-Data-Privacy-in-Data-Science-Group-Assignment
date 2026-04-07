"""FedNova and FedAvg strategies — extended with communication overhead tracking
and privacy-attack output logging (gradient vectors, update norms, participation).

New per-round logs
──────────────────
privacy_log  : one row per (round, client) — update norms + participation flags
               → written to  *_privacy.csv  by main.py
grad_log     : one entry per round — aggregated gradient vector as a numpy array
               → written to  *_grads.npz   by main.py

See bottom of file for the full "output → attack" mapping.
"""

from logging import INFO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    Metrics,
    NDArray,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from omegaconf import DictConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_params_and_bytes(ndarrays: NDArrays) -> Tuple[int, int]:
    """Return (total_parameters, total_bytes) for a list of numpy arrays."""
    total_params = sum(arr.size for arr in ndarrays)
    total_bytes = sum(arr.nbytes for arr in ndarrays)
    return total_params, total_bytes


def _build_comm_record(
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    global_params_bytes: int,
    global_params_count: int,
    cumulative_sent: int,
    cumulative_received: int,
) -> Tuple[Dict, int, int]:
    """Build a per-round communication record (one row per round).

    Returns the record dict plus updated cumulative sent/received totals.
    """
    num_clients = len(results)
    bytes_sent_this_round = global_params_bytes * num_clients

    bytes_received_this_round = 0
    for _client, fit_res in results:
        client_arrays = parameters_to_ndarrays(fit_res.parameters)
        _, c_bytes = _count_params_and_bytes(client_arrays)
        bytes_received_this_round += c_bytes

    cumulative_sent += bytes_sent_this_round
    cumulative_received += bytes_received_this_round

    record = {
        "round": server_round,
        "num_clients": num_clients,
        "global_model_params": global_params_count,
        "bytes_sent_per_client": global_params_bytes,
        "bytes_sent_total": bytes_sent_this_round,
        "bytes_received_total": bytes_received_this_round,
        "bytes_exchanged_total": bytes_sent_this_round + bytes_received_this_round,
        "cumulative_bytes_sent": cumulative_sent,
        "cumulative_bytes_received": cumulative_received,
        "cumulative_bytes_total": cumulative_sent + cumulative_received,
    }

    log(
        INFO,
        "[Round %d] Comm — sent: %.2f MB | received: %.2f MB | "
        "cumulative total: %.2f MB",
        server_round,
        bytes_sent_this_round / 1e6,
        bytes_received_this_round / 1e6,
        (cumulative_sent + cumulative_received) / 1e6,
    )

    return record, cumulative_sent, cumulative_received


def _build_privacy_records(
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    global_params: NDArrays,
) -> Tuple[List[Dict], NDArrays]:
    """Build per-client privacy records and the flattened aggregated update vector.

    For each participating client we record:
      - client_id        : Flower cid string
      - round            : server round number
      - participated     : always 1 here (only present clients are in results)
      - update_norm_l2   : L2 norm of the client's raw update vector
      - update_norm_l1   : L1 norm  (useful for sparsity analysis)
      - update_norm_linf : L-inf norm (max absolute weight change)
      - num_samples      : num_examples reported by the client

    Also returns the element-wise mean aggregated update (global_params - mean_client)
    as a flat 1-D numpy array, saved separately for gradient-inversion attacks.

    Notes on FedAvg vs FedNova
    ──────────────────────────
    FedAvg clients return model weights (not gradients).  The "update" is computed
    here as  (client_weights − global_weights), i.e. the pseudo-gradient.
    FedNova clients return cum_grad arrays directly — those are already gradient
    estimates, so the norm is taken directly.  The strategy cannot reliably
    distinguish the two at this level, so we apply the same delta computation
    for both; for FedNova the delta is (cum_grad − global_params), which is
    numerically large but consistently comparable across rounds and clients.
    """
    privacy_rows: List[Dict] = []

    # Flatten global params once for delta computation
    global_flat = np.concatenate([p.flatten() for p in global_params])

    client_flats: List[np.ndarray] = []
    for client, fit_res in results:
        client_arrays = parameters_to_ndarrays(fit_res.parameters)
        client_flat = np.concatenate([a.flatten() for a in client_arrays])
        client_flats.append(client_flat)

        # Delta = client update relative to global broadcast
        delta = client_flat - global_flat

        privacy_rows.append({
            "round": server_round,
            "client_id": client.cid,
            "participated": 1,
            "num_samples": fit_res.num_examples,
            "update_norm_l2": float(np.linalg.norm(delta, ord=2)),
            "update_norm_l1": float(np.linalg.norm(delta, ord=1)),
            "update_norm_linf": float(np.linalg.norm(delta, ord=np.inf)),
        })

    # Aggregated update vector: mean of all client deltas (flat, float32)
    # This is what an honest-but-curious server sees after aggregation.
    if client_flats:
        mean_client_flat = np.mean(np.stack(client_flats, axis=0), axis=0)
        agg_update_flat = (mean_client_flat - global_flat).astype(np.float32)
    else:
        agg_update_flat = np.zeros_like(global_flat, dtype=np.float32)

    return privacy_rows, agg_update_flat


# ── FedAvg with communication + privacy tracking ───────────────────────────────

class FedAvgWithCommTracking(FedAvg):
    """Standard FedAvg strategy with per-round communication and privacy tracking.

    Works for both FedAvg (mu=0) and FedProx (mu>0) — the proximal term
    lives in the client optimizer, not in the strategy.

    New public attributes
    ─────────────────────
    privacy_log : List[Dict]
        One row per (round × client).  Saved as *_privacy.csv by main.py.
    grad_log    : List[Dict]
        One entry per round with keys 'round' and 'agg_update' (flat np.ndarray).
        Saved as *_grads.npz by main.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.comm_log: List[Dict] = []
        self.cumulative_bytes_sent = 0
        self.cumulative_bytes_received = 0

        # ── NEW ──
        self.privacy_log: List[Dict] = []
        self.grad_log: List[Dict] = []

        if self.initial_parameters is not None:
            self._cached_global_params: NDArrays = parameters_to_ndarrays(
                self.initial_parameters
            )
        else:
            self._cached_global_params = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate client results and log communication + privacy stats."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        global_params_count, global_params_bytes = _count_params_and_bytes(
            self._cached_global_params
        )

        record, self.cumulative_bytes_sent, self.cumulative_bytes_received = (
            _build_comm_record(
                server_round,
                results,
                global_params_bytes,
                global_params_count,
                self.cumulative_bytes_sent,
                self.cumulative_bytes_received,
            )
        )
        self.comm_log.append(record)

        # ── NEW: privacy / gradient logging ───────────────────────────────────
        privacy_rows, agg_update_flat = _build_privacy_records(
            server_round, results, self._cached_global_params
        )
        self.privacy_log.extend(privacy_rows)
        self.grad_log.append({"round": server_round, "agg_update": agg_update_flat})

        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Update cached params for next round's broadcast count
        if aggregated_parameters is not None:
            self._cached_global_params = parameters_to_ndarrays(aggregated_parameters)

        return aggregated_parameters, aggregated_metrics


# ── FedNova with communication + privacy tracking ─────────────────────────────

class FedNova(FedAvg):
    """FedNova strategy with per-round communication and privacy tracking.

    New public attributes
    ─────────────────────
    privacy_log : List[Dict]   — same schema as FedAvgWithCommTracking
    grad_log    : List[Dict]   — same schema as FedAvgWithCommTracking
    """

    def __init__(self, exp_config: DictConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_momentum_buffer: List[NDArray] = []
        if self.initial_parameters is not None:
            self.global_parameters: List[NDArray] = parameters_to_ndarrays(
                self.initial_parameters
            )

        self.exp_config = exp_config
        self.lr = exp_config.optimizer.lr
        self.gmf = exp_config.optimizer.gmf
        self.best_test_acc = 0.0

        self.comm_log: List[Dict] = []
        self.cumulative_bytes_sent = 0
        self.cumulative_bytes_received = 0

        # ── NEW ──
        self.privacy_log: List[Dict] = []
        self.grad_log: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate client results and log communication + privacy stats."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        global_params_count, global_params_bytes = _count_params_and_bytes(
            self.global_parameters
        )

        record, self.cumulative_bytes_sent, self.cumulative_bytes_received = (
            _build_comm_record(
                server_round,
                results,
                global_params_bytes,
                global_params_count,
                self.cumulative_bytes_sent,
                self.cumulative_bytes_received,
            )
        )
        self.comm_log.append(record)

        # ── NEW: privacy / gradient logging ───────────────────────────────────
        privacy_rows, agg_update_flat = _build_privacy_records(
            server_round, results, self.global_parameters
        )
        self.privacy_log.extend(privacy_rows)
        self.grad_log.append({"round": server_round, "agg_update": agg_update_flat})

        # FedNova aggregation
        local_tau = [res.metrics["tau"] for _, res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []
        for _client, res in results:
            params = parameters_to_ndarrays(res.parameters)
            local_norm = float(res.metrics["local_norm"])
            if local_norm == 0:
                log(INFO, "Skipping client with local_norm=0 in FedNova aggregation.")
                continue
            scale = (tau_eff / local_norm) * float(res.metrics["weight"])
            aggregate_parameters.append((params, scale))

        agg_cum_gradient = aggregate(aggregate_parameters)
        self.update_server_params(agg_cum_gradient)

        return ndarrays_to_parameters(self.global_parameters), {}

    def update_server_params(self, cum_grad: NDArrays):
        """Update the global server parameters by aggregating client gradients."""
        for i, layer_cum_grad in enumerate(cum_grad):
            if self.gmf != 0:
                if len(self.global_momentum_buffer) < len(cum_grad):
                    buf = layer_cum_grad / self.lr
                    self.global_momentum_buffer.append(buf)
                else:
                    self.global_momentum_buffer[i] *= self.gmf
                    self.global_momentum_buffer[i] += layer_cum_grad / self.lr
                self.global_parameters[i] -= self.global_momentum_buffer[i] * self.lr
            else:
                self.global_parameters[i] -= layer_cum_grad

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Override default evaluate to save best model checkpoint."""
        if self.evaluate_fn is None:
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res
        accuracy = float(metrics["accuracy"])

        if accuracy > self.best_test_acc:
            self.best_test_acc = accuracy

            if server_round == 0:
                return None

            np.savez(
                f"{self.exp_config.checkpoint_path}bestModel_"
                f"{self.exp_config.exp_name}_{self.exp_config.strategy.name}_"
                f"varEpochs_{self.exp_config.var_local_epochs}.npz",
                global_parameters=np.array(self.global_parameters, dtype=object),
                metrics=np.array([loss, self.best_test_acc]),
                global_momentum_buffer=np.array(
                    self.global_momentum_buffer, dtype=object
                ),
            )
            log(INFO, "Model saved with Best Test accuracy %.3f: ", self.best_test_acc)

        return loss, metrics


# ── Metric aggregation ─────────────────────────────────────────────────────────

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics via weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": np.sum(accuracies) / np.sum(examples)}


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT → ATTACK MAPPING  (for reference)
# ══════════════════════════════════════════════════════════════════════════════
#
# FILE                          COLUMNS / KEYS              ATTACK(S)
# ─────────────────────────────────────────────────────────────────────────────
# *_privacy.csv
#   round                       int                         all
#   client_id                   str (Flower cid)            ➤ Aggregate Reconstruction
#   participated                1                           ➤ Aggregate Reconstruction
#   num_samples                 int                         ➤ Update Norm Inference
#   update_norm_l2              float                       ➤ Update Norm Inference
#                                                           ➤ Gradient Inversion (pre-filter)
#   update_norm_l1              float                       ➤ Update Norm Inference (sparsity)
#   update_norm_linf            float                       ➤ Update Norm Inference (outlier)
#
# *_grads.npz
#   round_<N>_agg_update        flat float32 ndarray        ➤ Gradient Inversion
#                                                           ➤ Aggregate Reconstruction
#
# *_comm.csv  (unchanged, already saved)
#   round, num_clients,
#   bytes_*                     ints                        ➤ Aggregate Reconstruction
#                                                             (participation count per round)
#
# partition_indices_*.json  (unchanged, already saved by dataset.py)
#   {client_id: [cifar_indices]}                            ➤ Membership Inference (ground truth)
#                                                           ➤ Update Norm Inference (data size)
#
# mia_outputs.py  (unchanged, run post-training)
#   probs, labels,
#   memberships, sample_indices                             ➤ Membership Inference
#
# ── What is still NOT available ────────────────────────────────────────────────
# DP config (ε, δ, C, σ)   → not applicable; no DP mechanism exists in this codebase.
#                             To enable DP Budget Analysis, integrate Opacus on the
#                             client side and log (noise_multiplier, max_grad_norm,
#                             delta, sample_rate) per round.
# ══════════════════════════════════════════════════════════════════════════════
