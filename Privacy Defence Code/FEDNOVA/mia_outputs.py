"""
Generate MIA (Membership Inference Attack) inputs from a saved FedNova/FedAvg/FedProx global model.

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
                --model_name SmallCNN \
                --partition   fednova/data/partition_indices_seed1_alpha0.1_clients16.json \
                --datapath    fednova/data/ \
                --output_path fednova/mia/mia_fednova_prox_fednova_seed1.npz
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import argparse
import json
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import fednova.models as fednova_models


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_model(model_name: str) -> torch.nn.Module:
    """
    Instantiate a model class from fednova.models by name.
    """
    if not hasattr(fednova_models, model_name):
        available = [
            name
            for name in dir(fednova_models)
            if not name.startswith("_") and isinstance(getattr(fednova_models, name), type)
        ]
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available classes: {', '.join(sorted(available))}"
        )

    model_cls = getattr(fednova_models, model_name)
    if not isinstance(model_cls, type) or not issubclass(model_cls, torch.nn.Module):
        raise ValueError(f"'{model_name}' is not a model class in fednova.models")
    try:
        return model_cls()
    except TypeError as err:
        raise ValueError(
            f"Model class '{model_name}' could not be instantiated without args: {err}"
        ) from err


def _list_model_classes() -> Dict[str, type]:
    """Return torch.nn.Module classes exposed from fednova.models."""
    model_classes: Dict[str, type] = {}
    for name in dir(fednova_models):
        if name.startswith("_"):
            continue
        value = getattr(fednova_models, name)
        if isinstance(value, type) and issubclass(value, torch.nn.Module):
            model_classes[name] = value
    return model_classes


def _model_matches_checkpoint(model_name: str, checkpoint_params: np.ndarray) -> bool:
    """Check if checkpoint parameter shapes match a model class exactly."""
    try:
        model = _build_model(model_name)
    except Exception:
        return False

    model_tensors = list(model.state_dict().values())
    if len(model_tensors) != len(checkpoint_params):
        return False

    for model_tensor, ckpt_value in zip(model_tensors, checkpoint_params):
        if tuple(model_tensor.shape) != tuple(np.asarray(ckpt_value).shape):
            return False

    return True


def _infer_model_name(checkpoint_params: np.ndarray) -> Tuple[str, List[str]]:
    """Infer model class name from checkpoint parameter shapes."""
    candidates: List[str] = []
    for model_name in sorted(_list_model_classes().keys()):
        if _model_matches_checkpoint(model_name, checkpoint_params):
            candidates.append(model_name)

    if not candidates:
        raise ValueError(
            "Could not infer model architecture from checkpoint shapes. "
            "Please pass --model_name explicitly."
        )

    return candidates[0], candidates


def load_model(model_path: str, model_name: str, device: torch.device) -> torch.nn.Module:
    """Load model weights from a .npz checkpoint into selected architecture."""
    checkpoint = np.load(model_path, allow_pickle=True)
    checkpoint_params = checkpoint["global_parameters"]

    selected_model_name = model_name
    if model_name.lower() == "auto":
        inferred_name, candidates = _infer_model_name(checkpoint_params)
        selected_model_name = inferred_name
        print(
            f"[MIA] Auto-detected model architecture: {inferred_name} "
            f"(candidates: {', '.join(candidates)})"
        )
    elif not _model_matches_checkpoint(model_name, checkpoint_params):
        try:
            inferred_name, candidates = _infer_model_name(checkpoint_params)
            hint = (
                f"Checkpoint does not match --model_name {model_name}. "
                f"Try --model_name {inferred_name}. "
                f"Matching candidates: {', '.join(candidates)}"
            )
        except ValueError:
            hint = (
                f"Checkpoint does not match --model_name {model_name}, and "
                "auto-inference found no exact match."
            )
        raise ValueError(hint)

    model = _build_model(selected_model_name)
    model_state = model.state_dict()
    if len(model_state) != len(checkpoint_params):
        raise ValueError(
            f"Parameter count mismatch for {selected_model_name}: "
            f"model has {len(model_state)} tensors, "
            f"checkpoint has {len(checkpoint_params)} tensors."
        )

    params_dict = zip(model_state.keys(), checkpoint_params)
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
    """Run inference and return (logits, probs, labels, indices)."""
    all_logits = []
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
            all_logits.append(logits.cpu().numpy())
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(target.numpy())
            all_indices.append(idx.numpy())

    return (
        np.concatenate(all_logits, axis=0),
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
        "--model_name", default="auto",
        help=(
            "Model class name from fednova.models "
            "(e.g., VGG, SmallCNN, ResNet20), or 'auto' to infer from checkpoint"
        )
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

    trainset_full = datasets.CIFAR10(
        root=args.datapath, train=True, download=False, transform=transform
    )
    testset = datasets.CIFAR10(
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
    print(f"[MIA] Loading model '{args.model_name}' from {args.model_path}")
    model = load_model(args.model_path, args.model_name, device)

    # ── Run inference ──────────────────────────────────────────────────────
    print("[MIA] Running inference on member samples (train set)...")
    member_logits, member_probs, member_labels, member_raw_idx = get_softmax_probs(
        model, member_loader, device
    )
    # member_raw_idx are indices into member_subset (0..len-1), remap to
    # original CIFAR-10 train indices for full traceability
    member_cifar_indices = np.array(member_indices)[member_raw_idx]

    print("[MIA] Running inference on non-member samples (test set)...")
    nonmember_logits, nonmember_probs, nonmember_labels, nonmember_raw_idx = get_softmax_probs(
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
        # Explicit split keys (for downstream scripts expecting train/test arrays)
        train_logits=member_logits,
        train_labels=member_labels,
        test_logits=nonmember_logits,
        test_labels=nonmember_labels,

        # Backward-compatible combined keys
        probs=probs,                   # shape (N, 10) — softmax outputs
        labels=labels,                 # shape (N,)    — true class labels
        memberships=memberships,       # shape (N,)    — 1=member, 0=non-member
        sample_indices=sample_indices, # shape (N,)    — original dataset indices
    )

    print(f"\n[MIA] Saved → {args.output_path}")
    print(f"      Members:     {memberships.sum()} samples")
    print(f"      Non-members: {(memberships == 0).sum()} samples")
    print(f"      train_logits shape: {member_logits.shape}")
    print(f"      test_logits shape:  {nonmember_logits.shape}")
    print(f"      probs shape: {probs.shape}")
    print(
        "\nYour group member can load this with:\n"
        "  data = np.load('mia_*.npz')\n"
        "  probs, memberships = data['probs'], data['memberships']"
    )


if __name__ == "__main__":
    main()
