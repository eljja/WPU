from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wpu.data.object_physics import ObjectPhysicsDataset, collate_physics_samples
from wpu.models.factory import create_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe whether event-frontier branch pooling fixes large-N WPU branch collapse."
    )
    parser.add_argument("--models", nargs="+", default=["wpu-sparse", "wpu-sparse-frontier", "serialized-token"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 23])
    parser.add_argument("--background-sizes", type=int, nargs="+", default=[80, 200, 400])
    parser.add_argument("--train-background-objects", type=int, default=200)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-csv", type=Path, default=Path("docs/experiments/large_n_frontier_pooling_probe.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/experiments/large_n_frontier_pooling_probe_results.md"))
    parser.add_argument("--out-ko-md", type=Path, default=Path("docs/experiments/large_n_frontier_pooling_probe_results.ko.md"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for model_name in args.models:
            model = _train_model(model_name, seed, args)
            model.eval()
            for background_objects in args.background_sizes:
                rows.append(_evaluate_model(model_name, seed, background_objects, model, args))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summary_rows(rows)
    args.out_md.write_text(_markdown_report(rows, summary, args, korean=False), encoding="utf-8")
    args.out_ko_md.write_text(_markdown_report(rows, summary, args, korean=True), encoding="utf-8")
    print(f"wrote={args.out_csv}")
    print(f"wrote={args.out_md}")
    print(f"wrote={args.out_ko_md}")


def _train_model(model_name: str, seed: int, args: argparse.Namespace) -> torch.nn.Module:
    torch.manual_seed(seed)
    dataset = ObjectPhysicsDataset(
        size=max(args.steps * args.batch_size, args.batch_size),
        seed=seed,
        background_objects=args.train_background_objects,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    model = create_model(model_name, hidden_dim=args.hidden_dim, layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    label_counts = Counter(dataset[index].branch_label for index in range(len(dataset)))
    class_weights = torch.tensor(
        [len(dataset) / max(1, label_counts.get(label, 0)) for label in range(3)],
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
        if step >= args.steps:
            break
    return model


def _evaluate_model(
    model_name: str,
    seed: int,
    background_objects: int,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> dict[str, object]:
    dataset = ObjectPhysicsDataset(size=args.eval_samples, seed=1000 + seed, background_objects=background_objects)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_physics_samples)
    total = 0
    correct = 0
    nll_total = 0.0
    mse_total = 0.0
    elapsed = 0.0
    with torch.no_grad():
        for batch, target_delta, labels in loader:
            start = time.perf_counter()
            prediction = model(batch, num_branches=3)
            elapsed += time.perf_counter() - start
            nll_total += float(F.cross_entropy(prediction.branch_logits, labels).item()) * labels.numel()
            mse_total += float(F.mse_loss(prediction.object_delta, target_delta).item()) * labels.numel()
            correct += int((prediction.branch_probabilities.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
    total_objects = background_objects + 4
    return {
        "model": model_name,
        "seed": seed,
        "background_objects": background_objects,
        "total_objects": total_objects,
        "branch_accuracy": round(correct / max(total, 1), 6),
        "branch_nll": round(nll_total / max(total, 1), 6),
        "mse": round(mse_total / max(total, 1), 6),
        "ms_per_sample": round((elapsed * 1000.0) / max(total, 1), 6),
        "work_proxy": _work_proxy(model_name, total_objects),
    }


def _work_proxy(model_name: str, total_objects: int) -> int:
    if model_name.startswith("wpu-"):
        return 3
    if model_name == "serialized-token":
        return (total_objects + 4) ** 2
    return total_objects * total_objects


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["total_objects"]))].append(row)
    summary = []
    for (model, total_objects), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        acc = [float(row["branch_accuracy"]) for row in values]
        nll = [float(row["branch_nll"]) for row in values]
        latency = [float(row["ms_per_sample"]) for row in values]
        summary.append(
            {
                "model": model,
                "total_objects": total_objects,
                "mean_accuracy": mean(acc),
                "ci95_accuracy": _ci95(acc),
                "mean_nll": mean(nll),
                "mean_ms_per_sample": mean(latency),
                "work_proxy": int(values[0]["work_proxy"]),
                "seeds": len(values),
            }
        )
    return summary


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / (len(values) ** 0.5)


def _markdown_report(
    rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    args: argparse.Namespace,
    *,
    korean: bool,
) -> str:
    if korean:
        title = "# Large-N Target/Frontier Pooling Probe"
        body = [
            "이 probe는 N>=204에서 보인 v1 WPU branch collapse의 한 원인이 global mean branch pooling인지 검사한다.",
            "수정 모델은 branch head가 전체 객체 평균이 아니라 event target 또는 one-hop relation frontier summary를 읽는다.",
            "이는 token으로 돌아가는 것이 아니라 event-conditioned causal working set을 branch decision의 native unit으로 쓰는 WPU식 수정이다.",
            "",
            f"설정: train background `{args.train_background_objects}`, steps `{args.steps}`, seeds `{args.seeds}`, eval total N `{[n + 4 for n in args.background_sizes]}`.",
            f"Source CSV: `{args.out_csv.as_posix()}`.",
            "",
            "## Aggregate Results",
            "",
            _summary_table(summary),
            "",
            "## Interpretation",
            "",
            "- 기존 `wpu-sparse`가 큰 N에서 흔들리면, 원인은 sparse propagation 자체만이 아니라 branch readout이 비인과 배경 객체에 희석되는 구조적 결함이다.",
            "- target/frontier WPU가 accuracy를 유지하면, WPU의 large-N 해법은 더 큰 dense attention이 아니라 event target/frontier에 대한 state-native readout이어야 한다.",
            "- 이 결과는 아직 물리 일반화의 해결이 아니다. causal working set이 커지거나 relation frontier가 틀리면 여전히 실패할 수 있다.",
        ]
    else:
        title = "# Large-N Target/Frontier Pooling Probe"
        body = [
            "This probe tests whether v1 WPU branch collapse at N>=204 is partly caused by global mean branch pooling.",
            "The patched models feed the branch head from the event target or one-hop relation frontier instead of the mean over all objects.",
            "This is not a return to token processing; it makes the event-conditioned causal working set the native readout unit.",
            "",
            f"Setup: train background `{args.train_background_objects}`, steps `{args.steps}`, seeds `{args.seeds}`, eval total N `{[n + 4 for n in args.background_sizes]}`.",
            f"Source CSV: `{args.out_csv.as_posix()}`.",
            "",
            "## Aggregate Results",
            "",
            _summary_table(summary),
            "",
            "## Interpretation",
            "",
            "- If the original `wpu-sparse` becomes unstable at large N, the cause is not only sparse propagation capacity; global branch readout dilutes causal state with non-causal objects.",
            "- If target/frontier WPU preserves accuracy, the large-N WPU fix is state-native target/frontier readout, not larger dense attention.",
            "- This does not solve broad physical generalization. WPU can still fail when the causal working set grows or relation frontiers are wrong.",
        ]
    return "\n".join([title, "", *body, "", "## Raw Rows", "", _raw_table(rows), ""])


def _summary_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| total N | model | mean accuracy | 95% CI | mean NLL | ms/sample | work proxy |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['total_objects']} | {row['model']} | {row['mean_accuracy']:.6f} | "
            f"{row['ci95_accuracy']:.6f} | {row['mean_nll']:.6f} | "
            f"{row['mean_ms_per_sample']:.6f} | {row['work_proxy']} |"
        )
    return "\n".join(lines)


def _raw_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| seed | total N | model | accuracy | NLL | MSE | ms/sample | work proxy |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['total_objects']} | {row['model']} | "
            f"{float(row['branch_accuracy']):.6f} | {float(row['branch_nll']):.6f} | "
            f"{float(row['mse']):.6f} | {float(row['ms_per_sample']):.6f} | {row['work_proxy']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
