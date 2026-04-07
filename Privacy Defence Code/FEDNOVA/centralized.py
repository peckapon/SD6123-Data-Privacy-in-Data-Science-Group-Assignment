"""
Centralized training baseline for CIFAR-10.

Trains a single model on the full CIFAR-10 training set.
Matches the FL experimental setup as closely as possible:
    - Same model family as the FL run
    - Same optimizer (SGD, lr=0.05, weight_decay=1e-4)
    - Same lr schedule (decay at 50% and 75% of rounds)
    - Same number of rounds (100), each round = 1 full epoch
    - Same batch size (32)
    - Same test set evaluation

Usage:
    python fednova/centralized.py
    python fednova/centralized.py --model smallcnn --num_rounds 100 --lr 0.05 --batch_size 32
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import argparse
from omegaconf import OmegaConf
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import SGD
from torchvision import datasets, transforms

from fednova.models import SmallCNN, VGG, test, train


def get_lr(base_lr: float, current_round: int, num_rounds: int) -> float:
    """
    Mirror the FL lr schedule: decay at 50% and 75% of rounds.
    """
    if current_round >= int(num_rounds * 0.75):
        return base_lr / 100
    elif current_round >= int(num_rounds / 2):
        return base_lr / 10
    return base_lr


def _init_early_stop_state(results_rows, warmup_rounds: int, min_delta: float):
    """
    Restore early-stop counters from prior centralized results.
    """
    best_loss = float("inf")
    best_round = 0
    bad_rounds = 0

    for row in sorted(results_rows, key=lambda item: item["round"]):
        round_num = int(row["round"])
        loss = float(row["test_loss"])
        if loss < (best_loss - min_delta):
            best_loss = loss
            best_round = round_num
            bad_rounds = 0
        elif round_num > warmup_rounds:
            bad_rounds += 1

    return best_loss, best_round, bad_rounds


def main():
    """
    Main entry point for centralized training baseline.
    Handles argument parsing, reproducibility, and device setup.
    """
    parser = argparse.ArgumentParser(description="Centralized CIFAR-10 baseline")
    parser.add_argument("--model", type=str, default="vgg", choices=["vgg", "smallcnn"])
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--datapath", type=str, default="fednova/data/")
    parser.add_argument("--checkpoint_path", type=str, default="fednova/checkpoints/")
    parser.add_argument("--output_path", type=str, default="fednova/results/")
    parser.add_argument("--early-stop-enabled", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stop-warmup-rounds", type=int, default=10)
    # DP arguments
    parser.add_argument("--dp-enabled", action="store_true", help="Enable DP training (Opacus)")
    parser.add_argument("--dp-noise-multiplier", type=float, default=1.0, help="Noise multiplier for DP")
    parser.add_argument("--dp-max-grad-norm", type=float, default=1.0, help="Max grad norm for DP")
    parser.add_argument("--dp-target-delta", type=float, default=1e-5, help="Target delta for DP epsilon computation")
    parser.add_argument("--dp-min-batch-size", type=int, default=8, help="Minimum batch size for DP retry")
    parser.add_argument("--dp-max-oom-retries", type=int, default=3, help="Max OOM retries for DP")
    args = parser.parse_args()

    # Try to load config from YAML if present
    config = None
    try:
        import os
        config_path = os.path.join(os.path.dirname(__file__), "conf", "base.yaml")
        if os.path.exists(config_path):
            config = OmegaConf.load(config_path)
    except Exception:
        config = None


    # Model selection: prefer config if present, else CLI
    model_name = args.model.lower()
    model_builder = None
    if config and hasattr(config, "model") and hasattr(config.model, "_target_"):
        # Use Hydra/OmegaConf _target_ for model
        import importlib
        target = config.model._target_
        module_name, class_name = target.rsplit('.', 1)
        model_module = importlib.import_module(module_name)
        model_builder = getattr(model_module, class_name)
        model_name = class_name.lower()
    else:
        model_builders = {
            "vgg": VGG,
            "smallcnn": SmallCNN,
        }
        model_builder = model_builders[model_name]

    # ── Reproducibility ────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data — same transforms as dataset.py ──────────────────────────────────
    # Training uses augmentation (random crop + flip), test does not.
    # This is identical to what FL clients see during local training.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root=args.datapath, train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root=args.datapath, train=False, download=True, transform=transform_test
    )

    # Full dataset in one loader — no partitioning, no heterogeneity
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False
    )

    # ── Model and optimizer ────────────────────────────────────────────────────
    model = model_builder().to(device)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # DP setup
    # Read DP config from YAML if available, else CLI
    dp_cfg = getattr(config, "dp", None) if config else None
    dp_enabled = bool((dp_cfg.enabled if dp_cfg else False) or getattr(args, "dp_enabled", False))
    dp_noise_multiplier = float(dp_cfg.noise_multiplier) if dp_cfg and hasattr(dp_cfg, "noise_multiplier") else getattr(args, "dp_noise_multiplier", 1.0)
    dp_max_grad_norm = float(dp_cfg.max_grad_norm) if dp_cfg and hasattr(dp_cfg, "max_grad_norm") else getattr(args, "dp_max_grad_norm", 1.0)
    dp_target_delta = float(dp_cfg.target_delta) if dp_cfg and hasattr(dp_cfg, "target_delta") else getattr(args, "dp_target_delta", 1e-5)
    dp_min_batch_size = int(dp_cfg.min_batch_size) if dp_cfg and hasattr(dp_cfg, "min_batch_size") else getattr(args, "dp_min_batch_size", 8)
    dp_max_oom_retries = int(dp_cfg.max_oom_retries) if dp_cfg and hasattr(dp_cfg, "max_oom_retries") else getattr(args, "dp_max_oom_retries", 3)
    if dp_enabled:
        from opacus import PrivacyEngine
        from opacus.grad_sample import GradSampleModule

    # ── Checkpoint resume ──────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    ckpt_path = os.path.join(
        args.checkpoint_path, f"checkpoint_centralized_{model_name}_seed{args.seed}.pt"
    )
    results_rows = []
    start_round = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        results_rows = ckpt["results_rows"]
        start_round = ckpt["completed_rounds"]
        print(f"[Checkpoint] Resuming from round {start_round}")

    best_test_loss, best_round, bad_rounds = _init_early_stop_state(
        results_rows,
        warmup_rounds=args.early_stop_warmup_rounds,
        min_delta=args.early_stop_min_delta,
    )

    if args.early_stop_enabled and bad_rounds >= args.early_stop_patience:
        print(
            f"[Early Stop] Already satisfied at round {start_round}. "
            f"No significant test-loss improvement since round {best_round}."
        )
        start_round = args.num_rounds

    # ── Training loop ──────────────────────────────────────────────────────────
    # Each round = 1 full epoch over the entire 50,000 sample training set.
    # This matches "num_rounds=100" in the FL setup, where each FL round
    # involves one round of local training across all participating clients.
    total_start = time.time()

    for round_num in range(start_round + 1, args.num_rounds + 1):
        round_start = time.time()

        # Apply lr schedule — mirrors fit_config() in utils.py
        current_lr = get_lr(args.lr, round_num, args.num_rounds)
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        # DP training logic (mirrors FL client logic)
        if dp_enabled:
            from torch.nn import CrossEntropyLoss
            from torch.cuda import OutOfMemoryError
            train_loss, train_acc = None, None
            dp_attempt_batches = []
            batch_size = args.batch_size
            min_batch_size = dp_min_batch_size
            max_oom_retries = dp_max_oom_retries
            # Build descending batch sizes for retry logic
            current_batch_size = batch_size
            while current_batch_size >= min_batch_size:
                dp_attempt_batches.append(current_batch_size)
                if current_batch_size == min_batch_size:
                    break
                current_batch_size = max(min_batch_size, current_batch_size // 2)
                if dp_attempt_batches and current_batch_size == dp_attempt_batches[-1]:
                    break
            if not dp_attempt_batches:
                dp_attempt_batches = [min_batch_size]
            max_attempts = min(len(dp_attempt_batches), max_oom_retries)
            for attempt_idx, batch_size in enumerate(dp_attempt_batches[:max_attempts], start=1):
                # Rebuild optimizer and model for each attempt
                optimizer = SGD(
                    model.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
                trainloader_dp = torch.utils.data.DataLoader(
                    trainset, batch_size=batch_size, shuffle=True, pin_memory=True
                )
                privacy_engine = PrivacyEngine()
                try:
                    # --- Opacus GradSampleModule unwrapping and hook cleanup ---
                    if 'GradSampleModule' in globals():
                        GradSampleModuleType = GradSampleModule
                    else:
                        from opacus.grad_sample import GradSampleModule as GradSampleModuleType
                    if isinstance(model, GradSampleModuleType):
                        model.disable_hooks()
                        model = model._module
                    # Remove any leftover hooks (from baseline_client.py)
                    for module_ in model.modules():
                        hook_handles = getattr(module_, "autograd_grad_sample_hooks", None)
                        if hook_handles is not None:
                            for handle in hook_handles:
                                try:
                                    handle.remove()
                                except Exception:
                                    pass
                            try:
                                delattr(module_, "autograd_grad_sample_hooks")
                            except Exception:
                                pass
                    model.train()  # Ensure model is in training mode for Opacus
                    # --- Ensure consistent loss_reduction for Opacus ---
                    from torch.nn import CrossEntropyLoss
                    criterion = CrossEntropyLoss(reduction="mean")
                    # Opacus make_private may return 3 or 4 values depending on version/args
                    private_components = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=trainloader_dp,
                        criterion=criterion,
                        noise_multiplier=dp_noise_multiplier,
                        max_grad_norm=dp_max_grad_norm,
                        loss_reduction="mean",
                        grad_sample_mode="ghost",
                    )
                    if isinstance(private_components, tuple) and len(private_components) == 4:
                        model_dp, optimizer_dp, criterion_dp, trainloader_dp = private_components
                    elif isinstance(private_components, tuple) and len(private_components) == 3:
                        model_dp, optimizer_dp, trainloader_dp = private_components
                        criterion_dp = criterion
                    else:
                        raise ValueError("Unexpected return value from PrivacyEngine.make_private")
                    train_loss, train_acc = train(
                        model_dp, optimizer_dp, trainloader_dp, device, epochs=1, criterion=criterion_dp
                    )
                    epsilon = privacy_engine.get_epsilon(delta=dp_target_delta)
                    hist = privacy_engine.accountant.history
                    noise_mult, sample_rate, dp_steps = (
                        hist[-1] if hist else (0.0, 0.0, 0)
                    )
                    # Log DP stats in results_rows
                    results_rows.append({
                        "round": round_num,
                        "lr": current_lr,
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "test_loss": None,  # Will be filled below
                        "test_accuracy": None,
                        "test_f1": None,
                        "test_precision": None,
                        "test_recall": None,
                        "dp_epsilon": float(epsilon),
                        "dp_noise_multiplier": float(noise_mult),
                        "dp_sample_rate": float(sample_rate),
                        "dp_num_steps": int(dp_steps),
                        "dp_batch_size": int(batch_size),
                        "dp_oom_retries": int(attempt_idx - 1),
                    })
                    break
                except OutOfMemoryError:
                    print(f"[DP] CUDA OOM during DP training (round {round_num}, batch_size={batch_size}).")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if attempt_idx < max_attempts:
                        print(f"[DP] Retrying DP training ({attempt_idx}/{max_attempts}).")
                        continue
                    # Log failed attempt
                    results_rows.append({
                        "round": round_num,
                        "lr": current_lr,
                        "train_loss": None,
                        "train_accuracy": None,
                        "test_loss": None,
                        "test_accuracy": None,
                        "test_f1": None,
                        "test_precision": None,
                        "test_recall": None,
                        "dp_epsilon": 0.0,
                        "dp_noise_multiplier": 0.0,
                        "dp_sample_rate": 0.0,
                        "dp_num_steps": 0,
                        "dp_batch_size": int(batch_size),
                        "dp_oom_retries": int(attempt_idx),
                        "dp_failed_oom": 1,
                    })
                    break
        else:
            # train() from models.py: 1 epoch over the full trainloader
            train_loss, train_acc = train(
                model, optimizer, trainloader, device, epochs=1
            )

        # test() from models.py: same evaluation used by FL strategies
        test_loss, metrics = test(model, testloader, device)

        elapsed = time.time() - round_start
        cumulative = time.time() - total_start

        # Update results for DP or append new for non-DP
        if dp_enabled and results_rows and results_rows[-1]["round"] == round_num:
            results_rows[-1].update({
                "test_loss": test_loss,
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "round_time_sec": round(elapsed, 2),
                "cumulative_time_sec": round(cumulative, 2),
            })
        else:
            results_rows.append({
                "round": round_num,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": test_loss,
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "round_time_sec": round(elapsed, 2),
                "cumulative_time_sec": round(cumulative, 2),
            })

        if test_loss < (best_test_loss - args.early_stop_min_delta):
            best_test_loss = test_loss
            best_round = round_num
            bad_rounds = 0
        elif round_num > args.early_stop_warmup_rounds:
            bad_rounds += 1

        print(
            f"[Round {round_num:3d}/{args.num_rounds}] "
            f"lr={current_lr:.5f} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | "
            f"test_loss={test_loss:.4f} | test_acc={metrics['accuracy']:.2f}%"
        )

        # Save checkpoint after every round (same pattern as main.py)
        torch.save({
            "completed_rounds": round_num,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "results_rows": results_rows,
        }, ckpt_path)

        if args.early_stop_enabled and round_num > args.early_stop_warmup_rounds:
            if bad_rounds >= args.early_stop_patience:
                print(
                    f"[Early Stop] Triggered at round {round_num}: "
                    f"no significant test-loss improvement for "
                    f"{args.early_stop_patience} round(s) since round {best_round}."
                )
                break

    # ── Save results CSV ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows).sort_values("round").reset_index(drop=True)
    csv_path = os.path.join(
        args.output_path, f"centralized_{model_name}_seed{args.seed}_results.csv"
    )
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # ── Save final model ───────────────────────────────────────────────────────
    final_model_path = os.path.join(
        args.checkpoint_path, f"finalModel_centralized_{model_name}_seed{args.seed}.npz"
    )
    np.savez(
        final_model_path,
        global_parameters=np.array(
            [v.cpu().numpy() for v in model.state_dict().values()], dtype=object
        ),
        test_loss=results_df["test_loss"].iloc[-1],
        test_accuracy=results_df["test_accuracy"].iloc[-1],
        test_f1=results_df["test_f1"].iloc[-1],
        test_precision=results_df["test_precision"].iloc[-1],
        test_recall=results_df["test_recall"].iloc[-1],
    )
    print(f"Final model saved → {final_model_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(
        f"\n{'='*60}\n"
        f"Centralized baseline complete ({model_name})\n"
        f"Rounds completed    : {int(results_df['round'].iloc[-1])}\n"
        f"Final test accuracy : {results_df['test_accuracy'].iloc[-1]:.4f}%\n"
        f"Final test F1       : {results_df['test_f1'].iloc[-1]:.4f}\n"
        f"Total time          : {total_elapsed / 60:.2f} minutes\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
