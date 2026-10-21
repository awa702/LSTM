"""Train the small floating-point LSTM baseline on prepared MIT-BIH windows."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class HeartbeatLSTM(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


def load_split(path: Path):
    item = np.load(path)
    return torch.from_numpy(item["x"]), torch.from_numpy(item["y"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_split(args.data_dir / "train.npz")
    valid_x, valid_y = load_split(args.data_dir / "validation.npz")
    test_x, test_y = load_split(args.data_dir / "test.npz")
    counts = torch.bincount(train_y, minlength=2).float()
    weights = counts.sum() / (2 * counts.clamp_min(1))
    model = HeartbeatLSTM(args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    loader = DataLoader(TensorDataset(train_x, train_y), args.batch_size, shuffle=True)
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred = model(valid_x.to(device)).argmax(1).cpu().numpy()
        score = f1_score(valid_y.numpy(), pred, average="macro", zero_division=0)
        print(f"epoch {epoch:02d} loss {np.mean(losses):.4f} val_macro_f1 {score:.4f}", flush=True)
        if score > best_f1:
            best_f1 = score
            torch.save({"model": model.state_dict(), "hidden": args.hidden}, args.out_dir / "model.pt")
    checkpoint = torch.load(args.out_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x.to(device)).argmax(1).cpu().numpy()
    result = {
        "device": str(device),
        "hidden": args.hidden,
        "best_validation_macro_f1": best_f1,
        "test_macro_f1": f1_score(test_y.numpy(), test_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(test_y.numpy(), test_pred).tolist(),
        "classification_report": classification_report(test_y.numpy(), test_pred, zero_division=0),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["classification_report"])


if __name__ == "__main__":
    main()
