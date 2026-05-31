# WPU 재현성 가이드

이 문서는 현재 저장소의 주장과 품질 검증을 재현하기 위한 최소 명령을 정리한다.
커밋된 evidence와, 검토 후 `docs/`로 승격해야 하는 generated artifact를 구분한다.

## 환경

권장 기본 환경:

- Python 3.11
- development dependencies 포함 editable install:

```bash
python -m pip install -e ".[dev]"
```

기본 설치는 standard PyTorch package를 함께 설치한다. CUDA 전용 실험은 로컬
driver/CUDA stack에 맞는 PyTorch build를 먼저 설치한 뒤 editable install을
실행한다.

Windows에서는 `python`이 실제 interpreter가 아니라 Microsoft Store alias로
잡힐 수 있다. 이런 경우 의도한 Python 설치로 virtual environment를 만든 뒤 venv
interpreter로 검증을 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

## 필수 품질 검증

논문, 문서, 모델 변경을 커밋하기 전에 다음을 실행한다.

```bash
python -m pytest
```

현재 test suite가 확인하는 항목:

- state model JSON과 delta overlay;
- scheduler behavior;
- sparse/dense/model shape path;
- rollout probability normalization;
- script entrypoint hygiene;
- README core script smoke execution;
- documentation link integrity;
- LaTeX figure와 citation integrity;
- experiment `Source CSV` integrity;
- robot-cup demo smoke output.

GitHub Actions는 push와 pull request에서 같은 test command를 실행한다.

## 데모 재현

```bash
python demos/robot_cup_demo.py
```

예상 출력 섹션:

- event와 initial frontier;
- scheduler와 model path;
- frontier trace;
- changed objects;
- branch probabilities;
- memory estimate.

## 논문 빌드

영문 PDF는 LaTeX source에서 생성한다.

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
```

한글 companion은 Markdown으로 관리한다.

```text
docs/arxiv/state_is_all_you_need_ko.md
```

## 실험 Artifact 정책

커밋된 paper evidence는 `docs/experiments/`와 `docs/figures/`에 둔다. 새 실험은
먼저 git ignore 대상인 `artifacts/`에 생성한다.

Generated result를 `docs/experiments/`로 승격하기 전 확인할 것:

- 의도한 모든 seed가 완료되었는지;
- matched-baseline claim을 위한 model parameter scale이 비교 가능한지;
- `Source CSV`가 포함되어 있고 비어 있지 않은지;
- 해석이 `docs/claims.ko.md`의 claim boundary를 따르는지;
- negative/mixed result를 숨기지 않았는지.

8M-class CWS GPU runner는 이 정책을 따른다.

```powershell
.\scripts\run_cws_8m_gpu.ps1 -Python python
```

출력은 승격 전 검토를 위해 `artifacts/causal_working_set_8m_gpu/`에 생성된다.

## 현재 제출 경계

`docs/claims.ko.md`를 authoritative claim boundary로 사용한다. 현재 저장소가
지지하는 것은 regime-specific WPU hypothesis이며, token/graph/latent world model
또는 hardware accelerator baseline에 대한 보편 우월성이 아니다.
