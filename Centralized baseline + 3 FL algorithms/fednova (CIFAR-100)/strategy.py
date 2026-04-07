"""FedNova and FedAvg strategies — extended with communication overhead tracking."""

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


# ── FedAvg with communication tracking ────────────────────────────────────────

class FedAvgWithCommTracking(FedAvg):
    """Standard FedAvg strategy with per-round communication tracking.

    Works for both FedAvg (mu=0) and FedProx (mu>0) — the proximal term
    lives in the client optimizer, not in the strategy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.comm_log: List[Dict] = []
        self.cumulative_bytes_sent = 0
        self.cumulative_bytes_received = 0

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
        """Aggregate client results and log per-round communication stats."""
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

        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Update cached params for next round's broadcast count
        if aggregated_parameters is not None:
            self._cached_global_params = parameters_to_ndarrays(aggregated_parameters)

        return aggregated_parameters, aggregated_metrics


# ── FedNova with communication tracking ───────────────────────────────────────

class FedNova(FedAvg):
    """FedNova strategy with per-round communication tracking."""

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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate client results and log per-round communication stats."""
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
