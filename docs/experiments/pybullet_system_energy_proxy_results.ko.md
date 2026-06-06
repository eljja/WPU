# PyBullet System Energy Proxy

이 문서는 실제 전력 측정이 아니라 systems profile에서 계산한 screening-only energy proxy다. 논문에서는 hardware evidence가 아니라 다음 측정 위치를 정하는 보조 지표로만 사용해야 한다.

Source CSVs:

- `docs/experiments/pybullet_system_profile.csv`
- `docs/experiments/pybullet_system_profile_cuda.csv`

Derived CSV:

- `docs/experiments/pybullet_system_energy_proxy.csv`

| profile | N | B | proxy reduction | latency reduction | memory reduction |
|---|---:|---:|---:|---:|---:|
| cpu_tensorization | 4.6 | 1 | 0.168570 | 0.225724 | 0.000000 |
| cpu_tensorization | 4.6 | 3 | 0.145088 | 0.194806 | 0.000000 |
| cpu_tensorization | 4.6 | 8 | 0.174860 | 0.195042 | 0.000000 |
| cpu_tensorization | 36.6 | 1 | 0.976404 | 0.831249 | 0.859694 |
| cpu_tensorization | 36.6 | 3 | 0.974745 | 0.820678 | 0.859694 |
| cpu_tensorization | 36.6 | 8 | 0.977935 | 0.840700 | 0.859694 |
| cpu_tensorization | 132.6 | 1 | 0.997873 | 0.945943 | 0.960764 |
| cpu_tensorization | 132.6 | 3 | 0.997973 | 0.948473 | 0.960764 |
| cpu_tensorization | 132.6 | 8 | 0.998048 | 0.950143 | 0.960764 |
| cpu_tensorization | 516.6 | 1 | 0.999851 | 0.985288 | 0.989891 |
| cpu_tensorization | 516.6 | 3 | 0.999857 | 0.985610 | 0.989891 |
| cpu_tensorization | 516.6 | 8 | 0.999849 | 0.985058 | 0.989891 |
| cpu_tensorization | 2052.6 | 1 | 0.999988 | 0.995233 | 0.997454 |
| cpu_tensorization | 2052.6 | 3 | 0.999989 | 0.995784 | 0.997454 |
| cpu_tensorization | 2052.6 | 8 | 0.999990 | 0.996035 | 0.997454 |
| cuda_forward_screening | 4.4 | 1 | 0.245737 | 0.244978 | 0.000000 |
| cuda_forward_screening | 4.4 | 3 | 0.239592 | 0.239443 | 0.000000 |
| cuda_forward_screening | 4.4 | 8 | 0.241216 | 0.241037 | 0.000000 |
| cuda_forward_screening | 36.4 | 1 | 0.820041 | 0.818833 | 0.006256 |
| cuda_forward_screening | 36.4 | 3 | 0.815622 | 0.814370 | 0.006256 |
| cuda_forward_screening | 36.4 | 8 | 0.817700 | 0.816534 | 0.006256 |
| cuda_forward_screening | 132.4 | 1 | 0.945591 | 0.944120 | 0.026337 |
| cuda_forward_screening | 132.4 | 3 | 0.945016 | 0.943537 | 0.026337 |
| cuda_forward_screening | 132.4 | 8 | 0.945194 | 0.943708 | 0.026337 |
| cuda_forward_screening | 516.4 | 1 | 0.986415 | 0.984922 | 0.098948 |
| cuda_forward_screening | 516.4 | 3 | 0.986425 | 0.984935 | 0.098948 |
| cuda_forward_screening | 516.4 | 8 | 0.986454 | 0.984966 | 0.098948 |
| cuda_forward_screening | 2052.4 | 1 | 0.997337 | 0.996173 | 0.304080 |
| cuda_forward_screening | 2052.4 | 3 | 0.997367 | 0.996216 | 0.304080 |
| cuda_forward_screening | 2052.4 | 8 | 0.997360 | 0.996206 | 0.304080 |

## Interpretation

- 최대 proxy reduction은 `0.999990`이며 profile `cpu_tensorization`, N `2052.6`, B `8`에서 발생한다.
- CUDA forward screening proxy의 최대 reduction은 `0.997367`이다.
- 이 값은 전력계 측정, GPU power telemetry, sparse kernel counter를 대체하지 못한다.
