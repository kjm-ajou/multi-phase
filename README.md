# 다중상 polymorph 경쟁 앱 — Streamlit 배포 가이드

prototype v3의 내용(전이율·경로 그래프 → 분율 경쟁 ODE → 물리시간 보정 phase field)을
Streamlit Community Cloud로 배포하는 방법입니다.

## 1. 필요한 파일 (repo 루트에 둘 것)

| 파일 | 역할 |
|------|------|
| `multiphase_app.py` | Streamlit UI (메인 파일) |
| `multiphase_core.py` | 계산 로직 (UI와 분리, import 됨) |
| `requirements.txt` | 의존 패키지 (이미 있다면 `pillow` 추가 필수 — gif 생성에 필요) |
| `model_core.py` | (선택) 2D two-step 엔진. 있으면 ℹ️ 탭에서 인식. 없어도 앱은 정상 동작 |

`requirements.txt`에는 최소한 다음이 있어야 합니다:
```
streamlit>=1.40
numpy>=1.26
scipy>=1.11
pandas>=2.0
matplotlib>=3.7
pillow>=10.0
```

## 2. 로컬에서 먼저 확인

```bash
pip install -r requirements.txt
streamlit run multiphase_app.py
```
브라우저가 열리면 세 탭(전이율·그래프 / 분율 ODE / phase field)에서 **계산 버튼**을 눌러 확인합니다.

## 3. GitHub에 올리기

기존 repo(`kjm-ajou/Two-step-nucleation`)에 파일을 추가:
```bash
git add multiphase_app.py multiphase_core.py requirements.txt
git commit -m "Add multiphase polymorph competition app (v3)"
git push
```

## 4. Streamlit Community Cloud 배포

기존 two-step 앱(`app.py`)을 건드리지 않는 **두 가지 방법** 중 택1.

### 방법 A — 같은 repo에서 두 번째 앱 (가장 간단, 권장)
Streamlit Cloud는 한 repo에서 메인 파일만 다르게 해 여러 앱을 배포할 수 있습니다.
1. https://share.streamlit.io 로그인 → **New app**.
2. Repository = `kjm-ajou/Two-step-nucleation`, Branch = `main`.
3. **Main file path = `multiphase_app.py`** 로 지정.
4. Deploy. 기존 two-step 앱과 별개의 URL이 생깁니다.

### 방법 B — 멀티페이지(한 앱에 두 페이지로 통합)
하나의 URL에서 두 도구를 모두 쓰고 싶다면:
1. `pages/` 폴더를 만들고 `multiphase_app.py`를 `pages/2_Polymorph_competition.py`로 복사.
   - 단, 멀티페이지에서는 `st.set_page_config(...)`를 페이지 파일 맨 위에 그대로 두면 됩니다.
2. 기존 `app.py`가 홈(첫 페이지)이 되고, 사이드바에 "Polymorph competition" 페이지가 추가됩니다.
3. 메인 파일은 `app.py` 그대로 두고 재배포(자동 반영).

> 참고: 방법 B로 옮길 때 파일명 앞의 숫자(`2_`)는 사이드바 정렬 순서를 정합니다.

## 5. 운영상 주의 (Community Cloud 자원 한계)

- 무료 티어는 RAM/CPU가 제한적입니다(약 1 GB). **phase field 탭이 가장 무겁습니다.**
- 기본값(nx=96, steps=300)은 보통 수십 초입니다. 느리거나 메모리 부족이면 탭에서 **nx·스텝 수를 줄이세요**(예: 64, 150).
- gif 생성은 추가 시간이 듭니다. 필요 없으면 "애니메이션(gif)도 생성" 체크를 해제.
- 앱이 일정 시간 미사용 시 sleep 상태가 되며, 다음 접속 시 자동으로 깨어납니다(첫 로드가 느릴 수 있음).

## 6. 무엇이 들어있나 / 한계 (정직한 구분)

- **전이율·그래프**: 각 내리막 전이의 1D master-equation 핵생성률 $K_{ab}$(Zeldovich 자동), 경로 bottleneck.
- **분율 경쟁 ODE**: 평균장 $\dot\phi=(\mathbf K+\mathbf G-\mathbf D)\phi$; 등온/냉각으로 우세상이 바뀜(예: 느린 냉각 → kinetic 선택).
- **phase field**: 다상 Allen–Cahn + $K$ 구동 시딩; **시간축은 Turnbull–Fisher 계면 속도에 맞춘 차수 수준의 물리 시간(ms)**.
- 파라미터는 Fe 스케일 **예시**이며 UI에서 모두 편집 가능. 정량 보정(σ(T), KWN, 간선별 2D 정밀화)은 다음 단계.

문제가 생기면 Streamlit Cloud의 **Manage app → Logs**에서 오류 메시지를 확인하세요.
