"""Export a trained PyTorch LSTM checkpoint as a documented C array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_baseline import HeartbeatLSTM


def c_array(name: str, values: np.ndarray) -> str:
    flat = values.reshape(-1)
    body = ",\n    ".join(", ".join(f"{float(x):.9g}f" for x in flat[i:i + 8])
                       for i in range(0, len(flat), 8))
    return f"static const float {name}[{len(flat)}] = {{\n    {body}\n}};\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/baseline/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline/model_export"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = HeartbeatLSTM(int(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    state = checkpoint["model"]
    manifest = {"hidden": int(checkpoint["hidden"]), "tensors": {}}
    source = "#include \"model_weights.h\"\n\n"
    header = "#pragma once\n\n#define LSTM_INPUT_SIZE 1\n"
    header += f"#define LSTM_HIDDEN_SIZE {checkpoint['hidden']}\n#define LSTM_OUTPUT_SIZE 2\n"
    for key, tensor in state.items():
        name = key.replace(".", "_")
        arr = tensor.detach().numpy().astype(np.float32)
        source += c_array(name, arr) + "\n"
        manifest["tensors"][key] = {"name": name, "shape": list(arr.shape), "elements": int(arr.size)}
        header += f"extern const float {name}[{arr.size}];\n"
    (args.output / "model_weights.c").write_text(source)
    (args.output / "model_weights.h").write_text(header)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    x = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32).reshape(1, 256, 1)
    with torch.no_grad():
        logits = model(x).numpy()[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
    np.savez(args.output / "golden_input.npz", x=x.numpy(), logits=logits, probabilities=probs)
    (args.output / "golden.json").write_text(json.dumps({
        "input": "linspace(-1, 1, 256)", "logits": logits.tolist(),
        "probabilities": probs.tolist(), "class": int(np.argmax(logits)),
    }, indent=2) + "\n")
    print("exported", args.output)


if __name__ == "__main__":
    main()
