# V1 Object Physics Results

## Commands

```bash
python -m pytest
python demos/robot_cup_demo.py
python scripts/eval_object_physics.py --samples 256 --batch-size 16 --seed 101 --checkpoint artifacts/nonexistent.pt
python scripts/train_object_physics.py --steps 200 --batch-size 16 --seed 13 --checkpoint artifacts/object_physics_weighted.pt
python scripts/eval_object_physics.py --samples 256 --batch-size 16 --seed 101 --checkpoint artifacts/object_physics_weighted.pt
python scripts/route_sweep.py --samples 24 --batch-size 8 --background-sizes 0 20 80
```

## Verification

```text
python -m pytest
7 passed
```

## Robot Cup Demo

```text
Event: robot hand touched cup
Initial frontier: cup_001
Scheduler path: sparse (rho=0.0000 < 0.05)
Model path: sparse
Frontier trace: [['cup_001'], ['table_001', 'hand_001', 'edge_001']]
Changed objects: ['cup_001', 'edge_001', 'hand_001', 'table_001']
```

## Untrained Evaluation

```text
checkpoint_loaded=False
samples=256
next_state_mse=0.8111
branch_nll=1.2070
branch_accuracy=0.1289
majority_baseline_accuracy=0.6680
label_counts={0: 171, 1: 33, 2: 52}
sparse_path_ratio=1.0000
hybrid_path_ratio=0.0000
dense_path_ratio=0.0000
```

## Trained Evaluation

```text
checkpoint_loaded=True
samples=256
next_state_mse=0.0005
branch_nll=0.8074
branch_accuracy=0.7188
majority_baseline_accuracy=0.6680
label_counts={0: 171, 1: 33, 2: 52}
sparse_path_ratio=1.0000
hybrid_path_ratio=0.0000
dense_path_ratio=0.0000
```

## Route Sweep

```text
background_objects=0 route_ratios={'sparse': 0.0, 'hybrid': 0.0, 'dense': 1.0}
background_objects=20 route_ratios={'sparse': 0.0, 'hybrid': 1.0, 'dense': 0.0}
background_objects=80 route_ratios={'sparse': 1.0, 'hybrid': 0.0, 'dense': 0.0}
```

## Meaning

The model learns the synthetic next-state delta rule reliably. Branch prediction
improves over both the untrained model and the majority baseline, but the margin
is still small because the task is imbalanced and simple.

The route sweep is the stronger WPU evidence. It shows that compute path is a
function of affected-state fraction: a local event in a tiny world is dense, the
same event in a medium world is hybrid, and the same event in a large world is
sparse.
