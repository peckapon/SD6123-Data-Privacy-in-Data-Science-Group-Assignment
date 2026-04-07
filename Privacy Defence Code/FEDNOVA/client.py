"""
Client implementation for FedNova.
Implements a Flower NumPyClient for federated learning.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fednova.models import test, train


class FedNovaClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """
    Standard Flower client for CNN training.
    Handles local training, DP batch sizing, and optimizer instantiation.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        net: torch.nn.Module,
        client_id: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        ratio: float,
        config: DictConfig,
    ):
        """
        Initialize the FedNova client with model, data, and config.
        Handles optimizer and DP-specific settings.
        """
        self.net = net
        self.exp_config = config

        if self.exp_config.var_local_epochs and (
            self.exp_config.exp_name == "proximal"
        ):
            # For only FedNova with proximal local solver and variable local epochs,
            # mu = 0.001 works best.
            # For other experiments, the default setting of mu = 0.005 works best
            # Ref: https://arxiv.org/pdf/2007.07481.pdf (Page 33, Section:
            # More Experiment Details)
            self.exp_config.optimizer.mu = 0.001

        self.optimizer = instantiate(
            self.exp_config.optimizer, params=self.net.parameters(), ratio=ratio
        )
        self.trainloader = trainloader
        self.valloader = valloader
        self.client_id = client_id
        self.device = device
        self.num_epochs = num_epochs
        self.data_ratio = ratio

    # ── Helper methods ─────────────────────────────────────────────────────────

    def _build_trainloader(self, batch_size: int) -> DataLoader:
        """
        Create a fresh trainloader with a specific physical batch size.
        """
        return DataLoader(
            self.trainloader.dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=bool(getattr(self.trainloader, "pin_memory", False)),
            num_workers=int(getattr(self.trainloader, "num_workers", 0)),
            drop_last=bool(getattr(self.trainloader, "drop_last", False)),
        )

    def _dp_batch_sizes(self) -> List[int]:
        """
        Return descending batch sizes to try for DP training.
        """
        configured_batch = getattr(self.exp_config.dp, "batch_size", None)
        start_batch_size = int(configured_batch or self.trainloader.batch_size or 1)
        min_batch_size = int(getattr(self.exp_config.dp, "min_batch_size", 1))

        batch_sizes: List[int] = []
        current_batch_size = start_batch_size
        while current_batch_size >= min_batch_size:
            batch_sizes.append(current_batch_size)
            if current_batch_size == min_batch_size:
                break
            current_batch_size = max(min_batch_size, current_batch_size // 2)
            if batch_sizes and current_batch_size == batch_sizes[-1]:
                break

        return batch_sizes

    def _max_dp_retries(self) -> int:
        """
        Return maximum number of DP fit attempts per round for this client.
        """
        return max(1, int(getattr(self.exp_config.dp, "max_oom_retries", 5)))

    def _dp_attempt_batches(self) -> List[int]:
        """
        Return per-attempt batch sizes, padded to max retry attempts.
        """
        batch_sizes = self._dp_batch_sizes()
        max_attempts = self._max_dp_retries()
        if not batch_sizes:
            return [1] * max_attempts
        if len(batch_sizes) >= max_attempts:
            return batch_sizes[:max_attempts]
        return batch_sizes + [batch_sizes[-1]] * (max_attempts - len(batch_sizes))

    def _cleanup_opacus_hooks(self) -> None:
        """
        Remove stale Opacus hooks left by failed/private wrapping attempts.
        """
        for module in self.net.modules():
            hook_handles = getattr(module, "autograd_grad_sample_hooks", None)
            if hook_handles is not None:
                for handle in hook_handles:
                    try:
                        handle.remove()
                    except Exception:  # pylint: disable=broad-except
                        pass
                try:
                    delattr(module, "autograd_grad_sample_hooks")
                except Exception:  # pylint: disable=broad-except
                    pass

    @staticmethod
    def _is_oom_error(err: BaseException) -> bool:
        """
        Return True if exception corresponds to CUDA OOM.
        """
        return "out of memory" in str(err).lower()

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net.

        Before the first optimizer step, state["cum_grad"] does not yet exist,
        so we fall back to returning the raw model parameters (zero gradients).
        """
        opt_state = self.optimizer.state_dict()["state"]
        if opt_state and "cum_grad" in next(iter(opt_state.values())):
            print(f"[DEBUG] Client {self.client_id}: returning cum_grad")
            return [
                val["cum_grad"].cpu().numpy()
                for _, val in opt_state.items()
            ]
        # Fallback: no optimizer state yet — return model parameters directly
        print(f"[DEBUG] Client {self.client_id}: FALLBACK returning model weights")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        self.optimizer.set_model_params(parameters)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        # Reset optimizer state each round to prevent actor memory growth.
        # Must be done BEFORE set_parameters so that set_model_params()
        # can repopulate old_init into the clean state dict.
        self.optimizer.state.clear()
        self.optimizer.local_normalizing_vec = 0
        self.optimizer.local_counter = 0
        self.optimizer.local_steps = 0

        self.set_parameters(parameters)
        self.optimizer.set_lr(config["lr"])

        if self.exp_config.var_local_epochs:
            seed_val = (
                2023
                + int(self.client_id)
                + int(config["server_round"])
                + int(self.exp_config.seed)
            )
            np.random.seed(seed_val)
            num_epochs = np.random.randint(
                self.exp_config.var_min_epochs, self.exp_config.var_max_epochs
            )
        else:
            num_epochs = self.num_epochs

        if self.exp_config.dp.enabled:
            from opacus import PrivacyEngine  # pylint: disable=import-outside-toplevel
            from opacus.grad_sample import GradSampleModule  # pylint: disable=import-outside-toplevel

            if isinstance(self.net, GradSampleModule):
                self.net.disable_hooks()
                self.net = self.net._module
            self._cleanup_opacus_hooks()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            dp_attempt_batches = self._dp_attempt_batches()
            max_attempts = len(dp_attempt_batches)
            for attempt_idx, batch_size in enumerate(dp_attempt_batches, start=1):
                # Reinitialize optimizer state and model params for each retry.
                self.optimizer.state.clear()
                self.optimizer.local_normalizing_vec = 0
                self.optimizer.local_counter = 0
                self.optimizer.local_steps = 0
                self.set_parameters(parameters)
                self.optimizer.set_lr(config["lr"])

                self._cleanup_opacus_hooks()
                privacy_engine = PrivacyEngine()
                model_dp: GradSampleModule | None = None
                criterion_dp = None
                try:
                    trainloader_dp = self._build_trainloader(batch_size)
                    base_criterion = torch.nn.CrossEntropyLoss(reduction="mean")

                    private_components = privacy_engine.make_private(
                        module=self.net,
                        optimizer=self.optimizer,
                        data_loader=trainloader_dp,
                        criterion=base_criterion,
                        noise_multiplier=self.exp_config.dp.noise_multiplier,
                        max_grad_norm=self.exp_config.dp.max_grad_norm,
                        loss_reduction="mean",
                        grad_sample_mode="ghost",
                    )

                    if len(private_components) == 4:
                        model_dp, optimizer_dp, criterion_dp, trainloader_dp = private_components
                    elif len(private_components) == 3:
                        model_dp, optimizer_dp, trainloader_dp = private_components
                    else:
                        raise ValueError(
                            "Unexpected return value from PrivacyEngine.make_private"
                        )

                    train(
                        model_dp,
                        optimizer_dp,
                        trainloader_dp,
                        self.device,
                        epochs=num_epochs,
                        criterion=criterion_dp,
                    )

                    epsilon = privacy_engine.get_epsilon(
                        delta=self.exp_config.dp.target_delta
                    )
                    hist = privacy_engine.accountant.history
                    noise_mult, sample_rate, dp_steps = (
                        hist[-1] if hist else (0.0, 0.0, 0)
                    )
                    grad_scaling_factor: Dict[str, float] = self.optimizer.get_gradient_scaling()
                    grad_scaling_factor.update({
                        "epsilon": float(epsilon),
                        "dp_noise_multiplier": float(noise_mult),
                        "dp_sample_rate": float(sample_rate),
                        "dp_num_steps": int(dp_steps),
                        "dp_batch_size": int(batch_size),
                        "dp_oom_retries": int(attempt_idx - 1),
                    })
                    return self.get_parameters({}), len(trainloader_dp), grad_scaling_factor

                except torch.cuda.OutOfMemoryError:
                    print(
                        f"[Client {self.client_id}] CUDA OOM during DP training "
                        f"(round {config.get('server_round', '?')}, batch_size={batch_size})."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if attempt_idx < max_attempts:
                        print(
                            f"[Client {self.client_id}] Retrying DP training "
                            f"({attempt_idx}/{max_attempts})."
                        )
                        continue

                except RuntimeError as err:
                    if self._is_oom_error(err):
                        print(
                            f"[Client {self.client_id}] Runtime CUDA OOM during DP training "
                            f"(round {config.get('server_round', '?')}, batch_size={batch_size})."
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if attempt_idx < max_attempts:
                            print(
                                f"[Client {self.client_id}] Retrying DP training "
                                f"({attempt_idx}/{max_attempts})."
                            )
                            continue
                    else:
                        raise

                finally:
                    if model_dp is not None and isinstance(model_dp, GradSampleModule):
                        model_dp.disable_hooks()
                        self.net = model_dp._module
                        del model_dp
                    self._cleanup_opacus_hooks()
                    del privacy_engine
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            print(
                f"[Client {self.client_id}] DP training exhausted {max_attempts} "
                f"attempts in round {config.get('server_round', '?')}. "
                "Returning zero-weight update."
            )
            return list(parameters), 0, {
                "weight": 0.0,
                "tau": 0.0,
                "local_norm": 0.0,
                "epsilon": 0.0,
                "dp_noise_multiplier": 0.0,
                "dp_sample_rate": 0.0,
                "dp_num_steps": 0,
                "dp_batch_size": 0,
                "dp_oom_retries": int(max_attempts),
                "dp_failed_oom": 1,
            }

        try:
            train(
                self.net, self.optimizer, self.trainloader, self.device, epochs=num_epochs
            )
        except torch.cuda.OutOfMemoryError:
            print(
                f"[Client {self.client_id}] CUDA OOM during non-DP training "
                f"(round {config.get('server_round', '?')}). Returning zero-weight update."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return list(parameters), 0, {
                "weight": 0.0,
                "tau": 0.0,
                "local_norm": 0.0,
                "failed_oom": 1,
            }
        except RuntimeError as err:
            if self._is_oom_error(err):
                print(
                    f"[Client {self.client_id}] Runtime CUDA OOM during non-DP training "
                    f"(round {config.get('server_round', '?')}). Returning zero-weight update."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return list(parameters), 0, {
                    "weight": 0.0,
                    "tau": 0.0,
                    "local_norm": 0.0,
                    "failed_oom": 1,
                }
            raise

        # Get ratio by which the strategy would scale local gradients from each client
        # We use this scaling factor to aggregate the gradients on the server
        grad_scaling_factor: Dict[str, float] = self.optimizer.get_gradient_scaling()

        return self.get_parameters({}), len(self.trainloader), grad_scaling_factor

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        # Evaluation ideally is done on validation set, but because we already know
        # the best hyper-parameters from the paper and since individual client
        # datasets are already quite small, we merge the validation set with the
        # training set and evaluate on the training set with the aggregated global
        # model parameters. This behaviour can be modified by passing the validation
        # set in the below test(self.valloader) function and replacing len(
        # self.valloader) below. Note that we evaluate on the centralized test-set on
        # server-side in the strategy.

        self.set_parameters(parameters)
        loss, metrics = test(self.net, self.trainloader, self.device)
        return float(loss), len(self.trainloader), metrics


def gen_clients_fednova(  # pylint: disable=too-many-arguments
    num_epochs: int,
    trainloaders: List[DataLoader],
    testloader: DataLoader,
    data_sizes: List,
    model: DictConfig,
    exp_config: DictConfig,
) -> Callable[[str], fl.client.Client]:
    """Return a generator function to create a FedNova client."""

    def client_fn(cid: str) -> fl.client.Client:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = instantiate(model)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        client_dataset_size: int = data_sizes[int(cid)]
        client_dataset_ratio: float = client_dataset_size / sum(data_sizes)

        return FedNovaClient(
            net,
            cid,
            trainloader,
            testloader,
            device,
            num_epochs,
            client_dataset_ratio,
            exp_config,
        ).to_client()

    return client_fn
