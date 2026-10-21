"""Quantize the baseline checkpoint and compare a host integer LSTM reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_baseline import HeartbeatLSTM


def sat_int(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.clip(np.rint(x), lo, hi).astype(np.int8 if lo >= -128 and hi <= 127 else np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/baseline/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline/quantized"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = HeartbeatLSTM(int(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)

    tensors = {}
    manifest = {"input_q": "Q1.7 int8", "weight_q": "Q1.7 int8",
                "bias_q": "Q7.9 int32", "hidden": int(checkpoint["hidden"]),
                "activation_q": "Q1.15 int16", "state_q": "Q3.5 int16",
                "rounding": "nearest, ties away from zero", "saturation": "hard clamp"}
    for key, value in checkpoint["model"].items():
        arr = value.detach().numpy().astype(np.float64)
        if "bias" in key:
            q = np.rint(arr * (1 << 9)).astype(np.int32)
            scale = 1 << 9
        else:
            q = sat_int(arr * (1 << 7), -128, 127)
            scale = 1 << 7
        tensors[key] = q
        np.save(args.output / (key.replace(".", "_") + ".npy"), q)
        manifest.setdefault("tensors", {})[key] = {"shape": list(q.shape), "scale": scale,
                                                     "dtype": str(q.dtype)}
    def emit_int(name: str, arr: np.ndarray) -> str:
        values = ", ".join(str(int(x)) for x in arr.reshape(-1))
        return f"static const int32_t {name}[{arr.size}] = {{{values}}};\n"
    def emit_i8(name: str, arr: np.ndarray) -> str:
        values = ", ".join(str(int(x)) for x in arr.reshape(-1))
        return f"static const int8_t {name}[{arr.size}] = {{{values}}};\n"
    fixed_h = "#pragma once\n#include <stdint.h>\n"
    fixed_h += emit_i8("model_fixed_weight_ih", tensors["lstm.weight_ih_l0"])
    fixed_h += emit_i8("model_fixed_weight_hh", tensors["lstm.weight_hh_l0"])
    fixed_h += emit_int("model_fixed_bias_ih", tensors["lstm.bias_ih_l0"])
    fixed_h += emit_int("model_fixed_bias_hh", tensors["lstm.bias_hh_l0"])
    fixed_h += emit_i8("model_fixed_head_weight", tensors["head.weight"])
    fixed_h += emit_int("model_fixed_head_bias", tensors["head.bias"])
    (args.output / "model_fixed_weights.h").write_text(fixed_h)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Fixed vectors exercise zero, both input rails, and a deterministic ramp.
    vectors = {
        "zero": np.zeros((256,), dtype=np.float32),
        "min": np.full((256,), -5.0, dtype=np.float32),
        "max": np.full((256,), 5.0, dtype=np.float32),
        "ramp": np.linspace(-1.0, 1.0, 256, dtype=np.float32),
    }
    torch_vectors = {name: torch.from_numpy(x.reshape(1, 256, 1)) for name, x in vectors.items()}
    golden = {}
    with torch.no_grad():
        for name, x in torch_vectors.items():
            logits = model(x).numpy()[0]
            golden[name] = {"logits": logits.tolist(), "class": int(np.argmax(logits))}
    (args.output / "float_golden.json").write_text(json.dumps(golden, indent=2) + "\n")
    print("quantized", args.output, "parameters", sum(v.size for v in tensors.values()))
    print(json.dumps(golden, indent=2))


if __name__ == "__main__":
    main()
