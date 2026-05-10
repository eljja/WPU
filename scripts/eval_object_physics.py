from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples
from wpu.engines.scheduler import ExecutionPath
from wpu.models.factory import MODEL_NAMES, create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--model", choices=MODEL_NAMES, default="wpu-routed")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--background-objects", type=int, default=80)
    parser.add_argument("--num-branches", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/object_physics.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = ObjectPhysicsDataset(size=args.samples, seed=args.seed, background_objects=args.background_objects)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    model = create_model(args.model, hidden_dim=args.hidden_dim)
    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if checkpoint.get("model", args.model) != args.model:
            raise ValueError(f"checkpoint model={checkpoint.get('model')} does not match --model={args.model}")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total = 0
    correct = 0
    nll_values: list[float] = []
    mse_values: list[float] = []
    path_counts = {path: 0 for path in ExecutionPath}
    label_counts: Counter[int] = Counter()
    with torch.no_grad():
        for batch, target_delta, branch_label in loader:
            prediction = model(batch, num_branches=args.num_branches)
            mse_values.append(float(F.mse_loss(prediction.object_delta, target_delta).item()))
            nll_values.append(float(F.cross_entropy(prediction.branch_logits, branch_label).item()))
            predicted = prediction.branch_probabilities.argmax(dim=-1)
            correct += int((predicted == branch_label).sum().item())
            total += int(branch_label.numel())
            label_counts.update(int(label) for label in branch_label.tolist())
            for path in prediction.selected_paths:
                path_counts[path] += 1

    majority = label_counts.most_common(1)[0][1] / max(total, 1) if label_counts else 0.0
    print(f"checkpoint_loaded={args.checkpoint.exists()}")
    print(f"model={args.model}")
    print(f"background_objects={args.background_objects}")
    print(f"num_branches={args.num_branches}")
    print(f"samples={total}")
    print(f"next_state_mse={sum(mse_values) / max(len(mse_values), 1):.4f}")
    print(f"branch_nll={sum(nll_values) / max(len(nll_values), 1):.4f}")
    print(f"branch_accuracy={correct / max(total, 1):.4f}")
    print(f"majority_baseline_accuracy={majority:.4f}")
    print(f"label_counts={dict(sorted(label_counts.items()))}")
    for path, count in path_counts.items():
        print(f"{path.value}_path_ratio={count / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
