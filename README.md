# ECG heartbeat classification on RV32I

This project trains and exports a small, deterministic LSTM for the MIT-BIH
Arrhythmia Database. The deployment target is a bare-metal RV32I CPU, so the
runtime model must not require floating point, Linux, a filesystem, or dynamic
allocation.

## First milestone

The model classifies one heartbeat at a time. A heartbeat is represented by a
fixed window around a detected R peak; the R peak is a sample position, not the
whole model input. The first experiment uses one lead (MLII), 360 Hz, and a
256-sample window (a candidate, not a frozen ADC interface). The initial task
is binary heartbeat classification, followed by the five AAMI-style groups
`N/S/V/F/Q`. The exact binary mapping must be recorded before training: AAMI
`N` includes bundle branch block beats, so `N` does not mean a medically normal
heart. `Q` means unknown/paced-related beats and must not silently become an
"abnormal" ground truth. An initial benchmark can use N versus S/V/F and
exclude Q/paced records explicitly.

The first implementation assumes that an offline preprocessor supplies the R
peak positions. This keeps dataset parsing, model training, quantization, and
RV32I verification independently testable. A real-time QRS/R-peak detector is
a later pipeline stage.

The first reproducible baseline uses class 0 for AAMI N (`N/L/R/e/j`) and class
1 for S/V/F (`A/a/J/S/V/E/F`); Q/unknown annotations are excluded. Each sample
is a 256-point MLII-centered window (128 samples on each side), standardized
with training-record mean and standard deviation and clipped to `[-5, 5]`.
These are baseline choices for the first experiment; the resulting split and
statistics are saved with the artifacts before quantization.

## Runtime and demo interpretation

One call to the LSTM consumes one window and returns one class (and optionally
the class scores). A continuous demo feeds windows at successive R peaks. The
display can show the scrolling ECG, the current R-peak marker, the current
class, and a running count or percentage over the last N beats. That running
summary is an application-level aggregation; it is not a second recurrent
model in this milestone.

## Model policy

We will train a small model from scratch with supervised labels. A public
repository or paper may guide preprocessing and architecture, but a generic
pretrained LSTM is not expected to transfer: its sampling rate, lead, window,
labels, and tensor layout would not match this deployment. The exported model
will have an explicit integer specification shared by the Python reference,
RV32I C implementation, and any future RTL accelerator.

## References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- PhysioNet 2017 single-lead rhythm challenge: https://physionet.org/content/challenge-2017/1.0.0/
- PTB-XL (larger multi-lead alternative): https://physionet.org/content/ptb-xl/1.0.3/
- Lightweight wearable ECG LSTM paper: https://arxiv.org/abs/1812.04818
- Compressed Bi-LSTM heartbeat-classification reference: https://arxiv.org/abs/2405.15312

MIT-BIH contains 48 half-hour two-channel recordings sampled at 360 Hz with
beat annotations. Splits must be patient-wise (records 201 and 202 belong to
the same subject); randomly splitting
individual beats would leak subject-specific morphology into the test set.

## Environment

Use the local virtual environment in `.venv`:

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -r requirements-gpu.txt
./.venv/bin/python scripts/check_environment.py
```

The host is WSL with Python 3.10 and an NVIDIA GeForce RTX 5060 GPU (8 GiB)
visible through the Windows driver. CUDA 12.8 PyTorch wheels include the
required user-space runtime; a separate Linux NVIDIA kernel driver or local
CUDA compiler is not needed for this training workflow. Keep the project on
the WSL Linux filesystem at `/home/gary/project/lstm`.

Verified on 2026-10-14: Python 3.10.12, PyTorch 2.11.0+cu128, CUDA runtime
12.8, and NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0).
The environment check passed an actual GPU LSTM forward pass, backward pass,
and Adam parameter update with input shape `[8, 256, 1]` and hidden size 16.
This is a synthetic environment check; no ECG model has been trained yet.

`requirements-lock.txt` records the installed environment after verification.
To reproduce all pinned packages in a fresh Python 3.10 virtual environment:

```sh
./.venv/bin/python -m pip install -r requirements-lock.txt
```

GPU training and RV32I integer inference are separate environments: PyTorch and
CUDA are only used on the training host.

## Current status

The preprocessing and training scripts are in `scripts/`. A first GPU run has
completed on a temporary 21-record subset (records 100--124 with 116/117
omitted while the proxy download is incomplete): 27,102 training windows,
5,657 validation windows, and 4,691 test windows. The hidden-16 model reached
validation macro-F1 0.527 and test macro-F1 0.492; its artifacts are in
`artifacts/baseline/`. The test set contains only 42 class-1 beats in this
temporary split, so this result is a pipeline check rather than a final model
claim.

For the formal experiment, complete MIT-BIH records must be downloaded and the
patient-wise split in `doc_lstm/npc_lstm_agent_handoff.md` regenerated. The
script checks `.hea`, `.atr`, and the expected `.dat` byte count before parsing:

```sh
./.venv/bin/python scripts/prepare_mitdb.py --download
./.venv/bin/python scripts/train_baseline.py --epochs 20 --hidden 16
```

The current network proxy returns intermittent 502/timeouts for large PhysioNet
files; the local complete records and temporary artifacts are retained so a
later download can resume without changing the training code.

## Deployment export (works with the temporary subset)

`scripts/export_model.py` converts a trained checkpoint into the first C
arrays and documentation needed for RV32I inference. It writes:

- `model_weights.c` / `model_weights.h`: raw float32 arrays with element order
  matching the PyTorch state dict (`ih`, `hh`, and biases). LSTM gate order is
  i/f/g/h inside each 4-hidden block.
- `manifest.json`: tensor shapes and element counts.
- `golden_input.npz` / `golden.json`: a fixed `linspace(-1, 1, 256)` input with
  float logits, probabilities, and class for later fixed-point/RV32I checks.

The hidden-16 checkpoint has 1250 parameters (5000 bytes in float32). This is
the floating-point export only; quantization has not started. It is regenerated
automatically after the formal dataset is complete and retrained.

The first quantization scaffold is now in `scripts/quantize_model.py` and
`runtime/lstm_fixed.c`. It exports Q1.7 int8 weights, Q7.9 int32 biases, and
fixed test vectors; the C reference compiles with
`-march=rv32i -mabi=ilp32`. Its sigmoid/tanh functions are deliberately
placeholders for interface and resource work, so this scaffold is not yet an
accuracy-approved fixed-point model.

### 64 KB memory estimate

For the current hidden-16 model, the measured RV32I object contains 1568 bytes
of code and 1640 bytes of quantized constants. The compiler reports a 272-byte
stack frame for `lstm_fixed_infer`; an int8 256-sample input buffer and two int16
logits add 260 bytes. The known core total is therefore about 3740 bytes before
the firmware's reset code, UART driver, buffers, and activation table. The
quantized model itself is only 1640 bytes; the 5000-byte float32 checkpoint is
not deployed on an RV32I CPU without floating point.

Even reserving several kilobytes for sigmoid/tanh tables, ADC double buffers,
UART state, and startup code leaves substantial room in 64 KB. The final link
map must still be checked after integrating the board firmware. Avoid pulling
in `printf` or a full dynamic runtime. Since RV32I has no multiply instruction,
`runtime/rv32i_mul.c` supplies the software `__mulsi3` helper used by the
integer reference.

The same model is also evaluated for RV32IM. With `-Os` and a minimal
freestanding entry point, the current reference links to approximately:

| target | code | model constants | static input/output | multiply support | core sections (stack excluded) |
|---|---:|---:|---:|---|---:|
| RV32I | 948 B | 1632 B (+8 B head bias folded into code) | 260 B | 72 B `__mulsi3` helper | 2840 B |
| RV32IM | 820 B | 1632 B (+8 B head bias folded into code) | 260 B | hardware `mul`, no helper | 2712 B |

Both are far below 64 KB. The M extension saves only about 123 bytes in this
small image; its main value is latency. A 256-step hidden-16 inference performs
about 290,848 scalar 32-bit products including gate, state-update, and output
operations. RV32IM executes these with `mul`; RV32I expands them through
software multiplication. Exact latency depends on the CPU implementation and
must be measured in the NPC rather than inferred from memory size. The measured
stack frame is 288 bytes at `-Os`, so the core plus this frame is about 3128 B
for RV32I or 3000 B for RV32IM, before the board firmware's own stack and
buffers.
