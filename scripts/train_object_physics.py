from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples
from wpu.models.factory import MODEL_NAMES, create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model", choices=MODEL_NAMES, default="wpu-routed")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--background-objects", type=int, default=80)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/object_physics.pt"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = ObjectPhysicsDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=args.seed,
        background_objects=args.background_objects,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    model = create_model(args.model, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    class_weights = torch.tensor(
        [
            len(dataset) / max(1, label_counts.get(label, 0))
            for label in range(3)
        ],
        dtype=torch.float32,
    )
    class_weights = class_weights / class_weights.mean()

    model.train()
    for step, (batch, target_delta, branch_label) in enumerate(loader, start=1):
        prediction = model(batch, num_branches=3)
        delta_loss = F.mse_loss(prediction.object_delta, target_delta)
        branch_loss = F.cross_entropy(prediction.branch_logits, branch_label, weight=class_weights)
        loss = delta_loss + branch_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0:
            print(f"step={step} loss={loss.item():.4f} delta={delta_loss.item():.4f} branch={branch_loss.item():.4f}")
        if step >= args.steps:
            break
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "model": args.model,
            "hidden_dim": args.hidden_dim,
            "background_objects": args.background_objects,
            "class_weights": class_weights,
            "label_counts": dict(label_counts),
        },
        args.checkpoint,
    )
    print(f"saved_checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
