from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from wpu.models.factory import MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--train-samples-bg", type=int, default=80)
    parser.add_argument("--eval-background-objects", type=int, nargs="+", default=[0, 20, 80])
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-seed", type=int, default=101)
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_suite.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for model_name in args.models:
        checkpoint = Path("artifacts") / f"{model_name}.pt"
        _run(
            [
                sys.executable,
                "scripts/train_object_physics.py",
                "--model",
                model_name,
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed),
                "--background-objects",
                str(args.train_samples_bg),
                "--checkpoint",
                str(checkpoint),
            ]
        )
        for background_objects in args.eval_background_objects:
            output = _run(
                [
                    sys.executable,
                    "scripts/eval_object_physics.py",
                    "--model",
                    model_name,
                    "--samples",
                    str(args.samples),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(args.eval_seed),
                    "--background-objects",
                    str(background_objects),
                    "--checkpoint",
                    str(checkpoint),
                ]
            )
            row = _parse_metrics(output)
            row["model"] = model_name
            row["eval_background_objects"] = str(background_objects)
            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.output}")


def _run(command: list[str]) -> str:
    print("running=" + " ".join(command))
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    print(completed.stdout)
    return completed.stdout


def _parse_metrics(output: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line and not line.startswith("label_counts="):
            key, value = line.split("=", 1)
            row[key.strip()] = value.strip()
    return row


if __name__ == "__main__":
    main()
