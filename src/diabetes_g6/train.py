from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score, roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from diabetes_g6.data import prepare_data
from diabetes_g6.models import BaselineMLP, DenoisingAutoencoderClassifier, MaskedFeatureMLP

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "raw" / "diabetes_large_dataset.csv"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_mask(x: torch.Tensor, groups: list[np.ndarray], rate: float) -> torch.Tensor:
    mask = torch.zeros_like(x, dtype=torch.bool)
    if rate <= 0:
        return mask
    for group in groups:
        hide = torch.rand((x.shape[0], 1), device=x.device) < rate
        mask[:, torch.as_tensor(group, device=x.device)] = hide
    return mask


def masked_reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(reconstruction.dtype)
    return (((reconstruction - target) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device()
    data = prepare_data(args.data, test_size=args.test_size, random_state=args.seed)
    print(f"Device: {device}")
    print(f"Train/test rows: {len(data.y_train):,}/{len(data.y_test):,}")
    print(f"Encoded input width: {data.x_train.shape[1]}")
    print(f"Clinical fields eligible for masking: {len(data.feature_groups)}")
    print(f"Excluded target-derived columns: {', '.join(data.dropped_leakage_columns) or 'none found'}")

    train_dataset = TensorDataset(
        torch.from_numpy(data.x_train), torch.from_numpy(data.y_train.astype(np.float32))
    )
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    input_dim = data.x_train.shape[1]

    if args.model == "dae":
        model: nn.Module = DenoisingAutoencoderClassifier(input_dim).to(device)
    elif args.model == "baseline":
        model = BaselineMLP(input_dim).to(device)
    else:
        model = MaskedFeatureMLP(input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    if args.model == "dae":
        model.train()
        for epoch in range(args.pretrain_epochs):
            total = 0.0
            for x, _ in loader:
                x = x.to(device)
                mask = sample_mask(x, data.feature_groups, args.train_mask_rate)
                _, reconstruction = model(x, mask)
                loss = masked_reconstruction_loss(reconstruction, x, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"Reconstruction epoch {epoch + 1}/{args.pretrain_epochs}: {total / max(1, len(loader)):.4f}")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            rate = 0.0 if args.model == "baseline" else args.train_mask_rate
            mask = sample_mask(x, data.feature_groups, rate)
            optimizer.zero_grad(set_to_none=True)
            if args.model == "dae":
                logits, reconstruction = model(x, mask)
                loss = bce(logits, y) + args.reconstruction_weight * masked_reconstruction_loss(reconstruction, x, mask)
            else:
                logits = model(x, mask)
                loss = bce(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Classification epoch {epoch + 1}/{args.epochs}: {total / max(1, len(loader)):.4f}")

    x_test = torch.from_numpy(data.x_test).to(device)
    y_test = data.y_test.astype(np.int64)
    model.eval()
    print("\nEvaluation by clinical-measurement masking rate")
    with torch.no_grad():
        for rate in (0.0, 0.1, 0.2, 0.4):
            mask = sample_mask(x_test, data.feature_groups, rate)
            if args.model == "dae":
                logits, _ = model(x_test, mask)
            else:
                logits = model(x_test, mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            predictions = (probabilities >= 0.5).astype(np.int64)
            metrics = {
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "pr_auc": average_precision_score(y_test, probabilities),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
            print(f"{rate:>4.0%} missing: " + "  ".join(f"{name}={value:.4f}" for name, value in metrics.items()))

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    checkpoint = output_dir / f"{args.model}.pt"
    torch.save({"model_state_dict": model.state_dict(), "input_dim": input_dim, "seed": args.seed}, checkpoint)
    print(f"Saved checkpoint: {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a diabetes classifier under missing clinical measurements.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", choices=("mlp", "dae", "baseline"), default="mlp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--train-mask-rate", type=float, default=0.2)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.train_mask_rate < 1:
        parser.error("--train-mask-rate must be in [0, 1).")
    if args.epochs < 1 or args.pretrain_epochs < 1:
        parser.error("Epoch counts must be positive.")
    train(args)


if __name__ == "__main__":
    main()
