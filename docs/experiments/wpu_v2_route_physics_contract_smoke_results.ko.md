# WPU v2 Route-Physics Contract Smoke

Source CSVs:

- `docs/experiments/wpu_v2_route_physics_contract_smoke_physics.csv`
- `docs/experiments/wpu_v2_route_physics_contract_smoke_state.csv`

이 smoke test는 P1/P4 input-contract 수정을 검증한다. 기존 route-regret physics context는
pair-distance 통계, target xy, event norm만 사용했다. 따라서 `catch_action` 같은
action-conditioned state와 `edge_distance`, `hand_distance`, `fall_risk`,
`angular_speed` 같은 physical object-state scalar가 `StateGraphBatch`까지 들어와도
route-regret decision 전에는 대부분 버려졌다.

수정 후 route-physics feature vector는 14개 scalar로 확장된다.

| group | fields |
|---|---|
| pair geometry | selected object pair distance의 minimum과 mean |
| target geometry | target x/y |
| target physical state | `edge_distance`, `hand_distance`, `fall_risk`, `angular_speed` |
| selected-set physical state | selected object들의 `edge_distance`, `hand_distance`, `fall_risk`, `angular_speed` 평균 |
| event action | `force`, `catch_action` |

## Smoke Results

| model | N | K | samples | route-regret corr | routed loss | sparse loss | dense compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-physics-regret-hybrid` | 128 | 8 | 24 | 0.015806 | 1.137126 | 1.137126 | 0.000000 |
| `wpu-cws-indexed-state-regret-hybrid` | 128 | 8 | 24 | -0.091476 | 1.102039 | 1.102039 | 0.000000 |

## Interpretation

이 결과는 accuracy improvement claim이 아니다. 확장된 state/action feature가 모델의
route-regret 경로에 연결된 뒤에도 학습 및 평가 루프가 정상 실행된다는 contract test다.
CPU small run이므로 성능 결론을 내리기에는 의도적으로 작다.

학문적 의미는 좁지만 중요하다. WPU가 state-native execution을 주장하려면 mechanism을
정의하는 state variable이 후속 routing layer에서 조용히 사라지면 안 된다. 이번 수정은
그 불일치를 하나 제거했고, 다음 P1/P4 실험이 objectified-state 정의와 더 잘 정렬되게
한다.

## Reproduction

```bash
python scripts/staged_regret_hybrid.py \
  --model-name wpu-cws-indexed-physics-regret-hybrid \
  --n-values 128 --k-values 8 --seeds 11 \
  --hidden-dim 32 --layers 1 --num-heads 4 --working-set-size 12 \
  --propagation-steps 4 --regret-steps 6 \
  --samples 24 --validation-samples 24 --batch-size 4 \
  --device cpu \
  --out docs/experiments/wpu_v2_route_physics_contract_smoke_physics.csv

python scripts/staged_regret_hybrid.py \
  --model-name wpu-cws-indexed-state-regret-hybrid \
  --n-values 128 --k-values 8 --seeds 11 \
  --hidden-dim 32 --layers 1 --num-heads 4 --working-set-size 12 \
  --propagation-steps 4 --regret-steps 6 \
  --samples 24 --validation-samples 24 --batch-size 4 \
  --device cpu \
  --out docs/experiments/wpu_v2_route_physics_contract_smoke_state.csv
```
