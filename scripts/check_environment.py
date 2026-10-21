"""Check dataset dependencies and exercise the intended GPU LSTM workload."""

import importlib
import platform

import torch


def main():
    print("Python:", platform.python_version())
    for name in ("numpy", "scipy", "wfdb", "sklearn", "matplotlib", "torch"):
        module = importlib.import_module(name)
        print(f"{name}: {module.__version__}")
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training was requested, but CUDA is unavailable")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Compiled architectures:", torch.cuda.get_arch_list())
    torch.manual_seed(42)
    lstm = torch.nn.LSTM(1, 16, batch_first=True).cuda()
    head = torch.nn.Linear(16, 2).cuda()
    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(head.parameters()))
    inputs = torch.randn(8, 256, 1, device="cuda")
    labels = torch.arange(8, device="cuda") % 2
    before = head.weight.detach().clone()
    optimizer.zero_grad()
    outputs, _ = lstm(inputs)
    loss = torch.nn.functional.cross_entropy(head(outputs[:, -1]), labels)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or torch.equal(before, head.weight):
        raise RuntimeError("LSTM optimization did not produce a valid parameter update")
    print(f"GPU LSTM forward/backward/optimizer: PASS (loss={loss.item():.6f})")


if __name__ == "__main__":
    main()
