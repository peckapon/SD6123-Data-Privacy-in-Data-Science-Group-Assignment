"""Centralized training baseline for CIFAR-10.

Trains a single VGG-A model on the full CIFAR-10 training set.
Matches the FL experimental setup as closely as possible:
  - Same model (VGG-A)
  - Same optimizer (SGD, lr=0.05, weight_decay=1e-4)
  - Same lr schedule (decay at 50% and 75% of rounds)
  - Same number of rounds (100), each round = 1 full epoch
  - Same batch size (32)
  - Same test set evaluation

Usage:
    python fednova/centralized.py
    python fednova/centralized.py --num_rounds 100 --lr 0.05 --batch_size 32
"""

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import SGD
from torchvision import datasets, transforms

from fednova.models import VGG, test, train


def get_lr(base_lr: float, current_round: int, num_rounds: int) -> float:
    """Mirror the FL lr schedule: decay at 50% and 75% of rounds."""
    if current_round >= int(num_rounds * 0.75):
        return base_lr / 100
    elif current_round >= int(num_rounds / 2):
        return base_lr / 10
    return base_lr


def main():
    parser = argparse.ArgumentParser(description="Centralized CIFAR-10 baseline")
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--datapath", type=str, default="fednova/data/")
    parser.add_argument("--checkpoint_path", type=str, default="fednova/checkpoints/")
    parser.add_argument("--output_path", type=str, default="fednova/results/")
    args = parser.parse_args()

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

    trainset = datasets.CIFAR100(
        root=args.datapath, train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR100(
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
    # Same VGG-A architecture used by all FL clients
    model = VGG().to(device)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # ── Checkpoint resume ──────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    ckpt_path = os.path.join(
        args.checkpoint_path, f"checkpoint_centralized_seed{args.seed}.pt"
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

        # train() from models.py: 1 epoch over the full trainloader
        train_loss, train_acc = train(
            model, optimizer, trainloader, device, epochs=1
        )

        # test() from models.py: same evaluation used by FL strategies
        test_loss, metrics = test(model, testloader, device)

        elapsed = time.time() - round_start
        cumulative = time.time() - total_start

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

    # ── Save results CSV ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows).sort_values("round").reset_index(drop=True)
    csv_path = os.path.join(
        args.output_path, f"centralized_seed{args.seed}_results.csv"
    )
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # ── Save final model ───────────────────────────────────────────────────────
    final_model_path = os.path.join(
        args.checkpoint_path, f"finalModel_centralized_seed{args.seed}.npz"
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
        f"Centralized baseline complete\n"
        f"Final test accuracy : {results_df['test_accuracy'].iloc[-1]:.4f}%\n"
        f"Final test F1       : {results_df['test_f1'].iloc[-1]:.4f}\n"
        f"Total time          : {total_elapsed / 60:.2f} minutes\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
