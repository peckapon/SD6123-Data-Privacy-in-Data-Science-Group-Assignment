"""Generate MIA inputs from a saved FedNova/FedAvg/FedProx global model.

Produces a single .npz file per experiment containing:
  - probs        : softmax probabilities, shape (N, 10)
  - labels       : ground-truth class labels, shape (N,)
  - memberships  : binary membership labels (1=member, 0=non-member), shape (N,)
  - sample_indices: original CIFAR-10 dataset indices, shape (N,)

Members  = all CIFAR-10 training samples that appear in at least one client partition
           (determined from the saved partition_indices JSON).
Non-members = the full CIFAR-10 test set (never seen during training).

Usage:
    python fednova/mia_outputs.py \
        --model_path  fednova/checkpoints/finalModel_fednova_prox_fednova_varEpoch_True_seed_1.npz \
        --partition   fednova/data/partition_indices_seed1_alpha0.1_clients16.json \
        --datapath    fednova/data/ \
        --output_path fednova/mia/mia_fednova_prox_fednova_seed1.npz
"""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fednova.models import VGG


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load VGG model weights from a .npz checkpoint."""
    checkpoint = np.load(model_path, allow_pickle=True)
    model = VGG()
    params_dict = zip(model.state_dict().keys(), checkpoint["global_parameters"])
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_softmax_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run inference and return (probs, labels, indices)."""
    all_probs = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for batch in loader:
            # DataLoader may return (data, label) or (data, label, index)
            # depending on whether the dataset tracks indices
            data, target, idx = batch
            data = data.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(target.numpy())
            all_indices.append(idx.numpy())

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_indices, axis=0),
    )


# ── Dataset with index tracking ────────────────────────────────────────────────

class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the sample's original index."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, label, idx


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate MIA inputs from saved model")
    parser.add_argument(
        "--model_path", required=True,
        help="Path to finalModel_*.npz or bestModel_*.npz"
    )
    parser.add_argument(
        "--partition", required=True,
        help="Path to partition_indices_*.json saved by dataset.py"
    )
    parser.add_argument(
        "--datapath", default="fednova/data/",
        help="Root directory for CIFAR-10 data"
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Where to save the MIA .npz output file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for inference (default: 256)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MIA] Using device: {device}")

    # ── Load partition indices ──────────────────────────────────────────────
    with open(args.partition) as f:
        partition_indices = json.load(f)

    # Flatten all client partitions into a single sorted list of member indices
    member_indices = sorted(
        set(idx for indices in partition_indices.values() for idx in indices)
    )
    print(f"[MIA] Total member samples: {len(member_indices)}")

    # ── Build transforms (no augmentation for inference) ───────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset_full = datasets.CIFAR100(
        root=args.datapath, train=True, download=False, transform=transform
    )
    testset = datasets.CIFAR100(
        root=args.datapath, train=False, download=False, transform=transform
    )

    # Subset to only the member indices (avoids including unpartitioned samples)
    member_subset = Subset(trainset_full, member_indices)

    member_loader = DataLoader(
        IndexedDataset(member_subset),
        batch_size=args.batch_size,
        shuffle=False,
    )
    nonmember_loader = DataLoader(
        IndexedDataset(testset),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[MIA] Loading model from {args.model_path}")
    model = load_model(args.model_path, device)

    # ── Run inference ──────────────────────────────────────────────────────
    print("[MIA] Running inference on member samples (train set)...")
    member_probs, member_labels, member_raw_idx = get_softmax_probs(
        model, member_loader, device
    )
    # member_raw_idx are indices into member_subset (0..len-1), remap to
    # original CIFAR-10 train indices for full traceability
    member_cifar_indices = np.array(member_indices)[member_raw_idx]

    print("[MIA] Running inference on non-member samples (test set)...")
    nonmember_probs, nonmember_labels, nonmember_raw_idx = get_softmax_probs(
        model, nonmember_loader, device
    )
    # non-member indices are already CIFAR-10 test set indices (0..9999)
    nonmember_cifar_indices = nonmember_raw_idx

    # ── Combine and save ───────────────────────────────────────────────────
    probs = np.concatenate([member_probs, nonmember_probs], axis=0)
    labels = np.concatenate([member_labels, nonmember_labels], axis=0)
    memberships = np.concatenate([
        np.ones(len(member_probs), dtype=np.int32),
        np.zeros(len(nonmember_probs), dtype=np.int32),
    ], axis=0)
    sample_indices = np.concatenate(
        [member_cifar_indices, nonmember_cifar_indices], axis=0
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.savez(
        args.output_path,
        probs=probs,                   # shape (N, 10) — softmax outputs
        labels=labels,                 # shape (N,)    — true class labels
        memberships=memberships,       # shape (N,)    — 1=member, 0=non-member
        sample_indices=sample_indices, # shape (N,)    — original dataset indices
    )

    print(f"\n[MIA] Saved → {args.output_path}")
    print(f"      Members:     {memberships.sum()} samples")
    print(f"      Non-members: {(memberships == 0).sum()} samples")
    print(f"      probs shape: {probs.shape}")
    print(
        "\nYour group member can load this with:\n"
        "  data = np.load('mia_*.npz')\n"
        "  probs, memberships = data['probs'], data['memberships']"
    )


if __name__ == "__main__":
    main()
