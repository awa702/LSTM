"""Download MIT-BIH and create a patient-wise heartbeat classification split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wfdb


NORMAL = set("NLR ej".replace(" ", ""))
ABNORMAL = set("AaJSVEF")
RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# Keep 201/202 together. The other records are assigned deterministically by
# shuffled record groups; this is a baseline split, not a clinical protocol.
TEST = {"105", "111", "116", "203", "221", "228", "234"}
VALID = {"108", "118", "124", "210", "215", "223", "232"}


def choose_lead(sig_names: list[str]) -> int:
    names = [name.upper() for name in sig_names]
    for preferred in ("MLII", "II"):
        if preferred in names:
            return names.index(preferred)
    return 0


def split_for(record: str) -> str:
    if record in TEST:
        return "test"
    if record in VALID:
        return "validation"
    return "train"


def record_complete(data_dir: Path, record: str) -> bool:
    header = data_dir / f"{record}.hea"
    annotation = data_dir / f"{record}.atr"
    data = data_dir / f"{record}.dat"
    if not header.exists() or not annotation.exists() or not data.exists():
        return False
    lines = header.read_text().splitlines()
    if len(lines) < 2:
        return False
    first = lines[0].split()
    format_fields = lines[1].split()
    if len(first) < 4 or len(format_fields) < 2:
        return False
    # MIT-BIH uses format 212: three bytes encode one frame of two signals.
    expected = 3 * int(first[3]) if format_fields[1].startswith("212") else None
    return expected is None or data.stat().st_size == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitdb"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--records", nargs="+", default=RECORDS,
                        help="record IDs to include; defaults to all MIT-BIH records")
    parser.add_argument("--download", action="store_true",
                        help="download selected records with WFDB before processing")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected_records = list(args.records)
    missing = [r for r in selected_records if not record_complete(args.data_dir, r)]
    if missing:
        if not args.download:
            raise FileNotFoundError(
                f"missing local record headers: {missing}; rerun with --download")
        # WFDB handles the PhysioNet record manifest and skips files already
        # present. A network proxy may make this take several minutes.
        wfdb.dl_database("mitdb", str(args.data_dir), records=selected_records,
                         keep_subdirs=False)
        missing = [r for r in selected_records if not record_complete(args.data_dir, r)]
        if missing:
            raise RuntimeError(f"records remain incomplete after download: {missing}")
    windows: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}
    labels: dict[str, list[int]] = {key: [] for key in windows}
    records: dict[str, list[str]] = {key: [] for key in windows}
    sum_x = sum_x2 = count = 0

    for record in selected_records:
        split = split_for(record)
        path = args.data_dir / record
        signal, fields = wfdb.rdsamp(str(path))
        lead = choose_lead(fields["sig_name"])
        annotation = wfdb.rdann(str(path), "atr")
        signal = signal[:, lead].astype(np.float32)
        for sample, symbol in zip(annotation.sample, annotation.symbol):
            if symbol in NORMAL:
                label = 0
            elif symbol in ABNORMAL:
                label = 1
            else:
                continue
            left, right = int(sample) - 128, int(sample) + 128
            if left < 0 or right > len(signal):
                continue
            window = signal[left:right]
            windows[split].append(window)
            labels[split].append(label)
            records[split].append(record)
            if split == "train":
                sum_x += float(window.sum())
                sum_x2 += float(np.square(window, dtype=np.float64).sum())
                count += window.size

    mean = sum_x / count
    std = max((sum_x2 / count - mean * mean) ** 0.5, 1e-8)
    metadata = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "sampling_hz": 360,
        "window_samples": 256,
        "left_samples": 128,
        "lead_policy": "MLII, fallback first lead",
        "normal_symbols": sorted(NORMAL),
        "abnormal_symbols": sorted(ABNORMAL),
        "excluded_symbols": "all others (including Q/paced/unknown)",
        "mean": mean,
        "std": std,
        "clip": [-5.0, 5.0],
        "splits": {key: sorted(set(records[key])) for key in records},
        "seed": 0,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for split in windows:
        x = np.asarray(windows[split], dtype=np.float32)
        x = np.clip((x - mean) / std, -5.0, 5.0)
        y = np.asarray(labels[split], dtype=np.int64)
        np.savez_compressed(args.out_dir / f"{split}.npz", x=x[:, :, None], y=y)
        print(split, "records", len(set(records[split])), "samples", len(y),
              "class_counts", np.bincount(y, minlength=2).tolist())
    print("metadata", args.out_dir / "metadata.json")


if __name__ == "__main__":
    main()
