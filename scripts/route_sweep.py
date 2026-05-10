from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples
from wpu.engines.scheduler import ExecutionPath
from wpu.models import WorldStateProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--background-sizes", type=int, nargs="+", default=[0, 20, 80])
    args = parser.parse_args()

    model = WorldStateProcessor()
    model.eval()
    for background_objects in args.background_sizes:
        counts = {path: 0 for path in ExecutionPath}
        dataset = ObjectPhysicsDataset(
            size=args.samples,
            seed=101,
            background_objects=background_objects,
        )
        total = 0
        for batch, _, _ in DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples):
            prediction = model(batch)
            total += len(prediction.selected_paths)
            for path in prediction.selected_paths:
                counts[path] += 1
        ratios = {path.value: counts[path] / max(total, 1) for path in counts}
        print(f"background_objects={background_objects} route_ratios={ratios}")


if __name__ == "__main__":
    main()
