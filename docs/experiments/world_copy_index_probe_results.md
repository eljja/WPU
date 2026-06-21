# World-Copy Index Probe

This probe checks whether a world-copy style WPU can keep the event-local causal working set `K` small as total `N` grows.
It is an index/substrate validation, not a trained accuracy benchmark.
Source CSV: `docs/experiments/world_copy_index_probe.csv`.

| total N | selected K | affected fraction | non-causal selected | selected objects |
|---:|---:|---:|---:|---|
| 104 | 4 | 0.03846154 | 0 | `cup table hand edge` |
| 1004 | 4 | 0.00398406 | 0 | `cup table hand edge` |
| 5004 | 4 | 0.00079936 | 0 | `cup table hand edge` |
| 10004 | 4 | 0.00039984 | 0 | `cup table hand edge` |
