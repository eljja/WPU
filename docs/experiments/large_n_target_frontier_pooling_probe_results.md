# Large-N Target/Frontier Pooling Probe

This probe tests whether v1 WPU branch collapse at N>=204 is partly caused by global mean branch pooling.
The patched models feed the branch head from the event target or one-hop relation frontier instead of the mean over all objects.
This is not a return to token processing; it makes the event-conditioned causal working set the native readout unit.

Setup: train background `200`, steps `80`, seeds `[13, 17, 23]`, eval total N `[84, 204, 404]`.
Source CSV: `docs/experiments/large_n_target_frontier_pooling_probe.csv`.

## Aggregate Results

| total N | model | mean accuracy | 95% CI | mean NLL | ms/sample | work proxy |
|---:|---|---:|---:|---:|---:|---:|
| 84 | serialized-token | 0.721354 | 0.022249 | 0.694329 | 0.200137 | 7744 |
| 84 | wpu-sparse | 0.473958 | 0.334353 | 1.035795 | 0.476059 | 3 |
| 84 | wpu-sparse-frontier | 0.781250 | 0.038535 | 0.597070 | 0.532431 | 3 |
| 84 | wpu-sparse-target | 0.760416 | 0.005104 | 0.609870 | 0.475308 | 3 |
| 204 | serialized-token | 0.773438 | 0.008841 | 0.551877 | 0.448915 | 43264 |
| 204 | wpu-sparse | 0.643229 | 0.005104 | 1.032121 | 0.502622 | 3 |
| 204 | wpu-sparse-frontier | 0.781250 | 0.038535 | 0.597070 | 0.541357 | 3 |
| 204 | wpu-sparse-target | 0.760416 | 0.005104 | 0.609870 | 0.491859 | 3 |
| 404 | serialized-token | 0.778646 | 0.005104 | 0.583367 | 1.330520 | 166464 |
| 404 | wpu-sparse | 0.359375 | 0.275767 | 1.083916 | 0.543259 | 3 |
| 404 | wpu-sparse-frontier | 0.781250 | 0.038535 | 0.597070 | 0.586796 | 3 |
| 404 | wpu-sparse-target | 0.760416 | 0.005104 | 0.609870 | 0.541370 | 3 |

## Interpretation

- If the original `wpu-sparse` becomes unstable at large N, the cause is not only sparse propagation capacity; global branch readout dilutes causal state with non-causal objects.
- If target/frontier WPU preserves accuracy, the large-N WPU fix is state-native target/frontier readout, not larger dense attention.
- This does not solve broad physical generalization. WPU can still fail when the causal working set grows or relation frontiers are wrong.

## Raw Rows

| seed | total N | model | accuracy | NLL | MSE | ms/sample | work proxy |
|---:|---:|---|---:|---:|---:|---:|---:|
| 13 | 84 | wpu-sparse | 0.648438 | 0.950047 | 0.006425 | 0.458238 | 3 |
| 13 | 204 | wpu-sparse | 0.648438 | 1.007729 | 0.003844 | 0.505181 | 3 |
| 13 | 404 | wpu-sparse | 0.210938 | 1.114698 | 0.006298 | 0.543643 | 3 |
| 13 | 84 | wpu-sparse-target | 0.757812 | 0.650952 | 0.007981 | 0.465195 | 3 |
| 13 | 204 | wpu-sparse-target | 0.757812 | 0.650952 | 0.003888 | 0.481045 | 3 |
| 13 | 404 | wpu-sparse-target | 0.757812 | 0.650952 | 0.005092 | 0.564063 | 3 |
| 13 | 84 | wpu-sparse-frontier | 0.742188 | 0.661557 | 0.010164 | 0.522627 | 3 |
| 13 | 204 | wpu-sparse-frontier | 0.742188 | 0.661557 | 0.004794 | 0.534575 | 3 |
| 13 | 404 | wpu-sparse-frontier | 0.742188 | 0.661557 | 0.005693 | 0.575409 | 3 |
| 13 | 84 | serialized-token | 0.718750 | 0.721107 | 0.011280 | 0.212221 | 7744 |
| 13 | 204 | serialized-token | 0.773438 | 0.548802 | 0.002105 | 0.446938 | 43264 |
| 13 | 404 | serialized-token | 0.773438 | 0.554604 | 0.004630 | 1.585607 | 166464 |
| 17 | 84 | wpu-sparse | 0.640625 | 0.973157 | 0.003684 | 0.474678 | 3 |
| 17 | 204 | wpu-sparse | 0.640625 | 1.052373 | 0.002340 | 0.494405 | 3 |
| 17 | 404 | wpu-sparse | 0.226562 | 1.235393 | 0.004369 | 0.551758 | 3 |
| 17 | 84 | wpu-sparse-target | 0.757812 | 0.581647 | 0.006699 | 0.474211 | 3 |
| 17 | 204 | wpu-sparse-target | 0.757812 | 0.581647 | 0.003320 | 0.502545 | 3 |
| 17 | 404 | wpu-sparse-target | 0.757812 | 0.581647 | 0.004117 | 0.529560 | 3 |
| 17 | 84 | wpu-sparse-frontier | 0.804688 | 0.563035 | 0.010790 | 0.534377 | 3 |
| 17 | 204 | wpu-sparse-frontier | 0.804688 | 0.563035 | 0.005010 | 0.561155 | 3 |
| 17 | 404 | wpu-sparse-frontier | 0.804688 | 0.563035 | 0.005115 | 0.589213 | 3 |
| 17 | 84 | serialized-token | 0.703125 | 0.729257 | 0.021280 | 0.191677 | 7744 |
| 17 | 204 | serialized-token | 0.781250 | 0.568697 | 0.003061 | 0.463073 | 43264 |
| 17 | 404 | serialized-token | 0.781250 | 0.628661 | 0.011373 | 1.225594 | 166464 |
| 23 | 84 | wpu-sparse | 0.132812 | 1.184180 | 0.005128 | 0.495260 | 3 |
| 23 | 204 | wpu-sparse | 0.640625 | 1.036261 | 0.003167 | 0.508279 | 3 |
| 23 | 404 | wpu-sparse | 0.640625 | 0.901657 | 0.007554 | 0.534377 | 3 |
| 23 | 84 | wpu-sparse-target | 0.765625 | 0.597011 | 0.005687 | 0.486519 | 3 |
| 23 | 204 | wpu-sparse-target | 0.765625 | 0.597011 | 0.003291 | 0.491987 | 3 |
| 23 | 404 | wpu-sparse-target | 0.765625 | 0.597011 | 0.008281 | 0.530487 | 3 |
| 23 | 84 | wpu-sparse-frontier | 0.796875 | 0.566619 | 0.007377 | 0.540290 | 3 |
| 23 | 204 | wpu-sparse-frontier | 0.796875 | 0.566619 | 0.003998 | 0.528342 | 3 |
| 23 | 404 | wpu-sparse-frontier | 0.796875 | 0.566619 | 0.008439 | 0.595766 | 3 |
| 23 | 84 | serialized-token | 0.742188 | 0.632623 | 0.024754 | 0.196512 | 7744 |
| 23 | 204 | serialized-token | 0.765625 | 0.538132 | 0.001870 | 0.436735 | 43264 |
| 23 | 404 | serialized-token | 0.781250 | 0.566837 | 0.005137 | 1.180359 | 166464 |
