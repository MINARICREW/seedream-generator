## Seedream 4.0 로컬 이미지 스튜디오 (Streamlit)

바이트댄스 Seedream 4.0 이미지 생성/편집을 로컬에서 빠르게 실험·비교·관리할 수 있는 스트림릿 앱입니다. 대량 프롬프트/그룹 매트릭스(Grid), 폴더 벌크 실행, 세션 저장/복원, 라이브 로그/진행률, 병렬 실행/스트리밍 등 실제 작업 흐름에 필요한 기능을 모두 담았습니다.

---

### 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

환경 변수 설정(`env.example` → `env.local` 복사 후 편집):

- `ARK_API_KEY` (필수): Ark Console API Key
- `SEEDREAM_MODEL_ID` (선택): 모델 ID 또는 엔드포인트 ID (기본 `seedream-4-0-250828`)
- `OUTPUT_BASE_DIR` (선택): 결과물 기본 저장 위치 (기본 `outputs`)
- `MAX_PARALLEL` (선택): Grid/Bulk 기본 동시 요청 수 (기본 `8`)
- `STREAMING` (deprecated): 앱에서 스트리밍을 항상 사용하므로 이 값은 무시됩니다

앱 실행:

```bash
streamlit run app.py
```

---

### 전체 기능 개요

#### 사이드바(공통)
- **Settings**: 상단 배너(⚙️ 아이콘) – Output base folder, Model ID
- **Performance**: Max parallel requests(1~16). 스트리밍은 항상 활성화(토글 없음)
- **Progress**: Grid/Bulk 진행률 (현재/전체)
- **Live Logs**: 업로드/전처리/요청/저장/에러 등 이벤트를 실시간 표시

#### 탭 구성

1) ### Text to Image (T2I)
- 프롬프트 입력 → **Aspect ratio + Preset(1K/2K/4K)**로 해상도 자동 설정(필요 시 `WxH` 수동 조정 가능)
- 비율 라벨: `7:9 – 주민등록증/여권사진`, `5:6 – 증명사진`, `5:7 – 명함판`, `3:4 – 반명함`
- 필요 시 배치 모드 선택(Sequential `auto` 또는 `disabled`)
- 결과는 자동 저장(`metadata.json` 포함), 미리보기는 너비 500px로 균일 표시
- 병렬 설정이 그대로 적용. 모델이 스트리밍을 지원하면 중간 이미지가 자동으로 표시됩니다.

2) ### Image to Image (I2I)
- 1~10장의 **원본** 이미지를 업로드하여 프롬프트와 함께 전송
- 전처리/크랍/압축/포맷변경 없음(그대로 API로 전달)
- 해상도는 **Aspect ratio + Preset(1K/2K/4K)**로 지정(필요 시 `WxH` 수동 조정 가능)
- 결과는 자동 저장, 업로드한 레퍼런스는 세션 폴더 `references/`에도 복사

3) ### Grid Generator (멀티 그룹 × 멀티 프롬프트)

대량 비교·생성 워크플로우를 위한 핵심 기능입니다.

- #### 공통 프롬프트(상단)
  - 최대 10개까지 행(row)을 추가하여 입력
  - **프롬프트 변수** 지원: `{gender}` 같은 형태를 자동 감지 → 각 그룹 탭에 해당 변수 입력란 생성 → 그룹별 값으로 치환 후 호출

- #### 그룹 로드
  - **Load groups from folder (optional)**: 예) `inputs`를 지정하면 하위 1단계 폴더들이 그룹으로 자동 등록(최대 10개)
  - 그룹명은 폴더명, 그룹 이미지들은 해당 폴더 **바로 아래**의 파일만 사용(더 깊은 하위 폴더는 무시)
  - HEIC/HEIF 포함(.jpg/.jpeg/.png/.webp/.heic/.heif)

- #### 그룹 수 입력
  - 폴더에서 그룹을 로드하면 상단 `Number of reference groups`는 로드된 그룹 수로 자동 설정되며 입력은 비활성화됩니다.
  - 폴더 로드가 없는 상태에서는 수동으로 조절할 수 있습니다.

- #### 그룹 탭
  - 그룹명 변경, 그룹 변수 입력, 그룹 이미지 업로드(폴더 로드 안 한 경우)
  - 썸네일은 2컬럼으로 가득 채워 표시

- #### 사이즈/비율
  - Grid도 **Aspect ratio + Preset(1K/2K/4K)**로 `Width × Height`가 자동 설정됩니다.
  - 필요 시 자동 설정값을 기준으로 `WxH`가 정렬되어 요청에 사용됩니다.

- #### 공통 레퍼런스(상단)
  - 최대 **5장**까지 업로드 가능
  - **전처리 없이 원본으로 전송**, 각 그룹 요청 시 **이미지 인덱스 1..N**을 공통으로 차지
  - API 제약(총 레퍼런스 ≤ 10)을 자동 반영: 공통이 N장이면 그룹별 레퍼런스는 최대 `10−N`장만 전송

- #### DNN 얼굴 크랍(Balanced Crop v2, 선택)
  - 체크 시 그룹 이미지에만 적용(공통 레퍼런스/I2I에는 미적용)
  - `cropper_v2.py`의 `FaceCropper`를 동적으로 로드하여 DNN 기반 박스 검출 → Balanced Crop v2 → **정사각 1024×1024** 리사이즈
  - 검출 실패 시 중앙 안전 크랍 폴백

- #### 실행/저장/복원
  - Grid는 **병렬 실행**과 **스트리밍**(모델 지원 시, 앱에서 항상 활성화)을 통해 셀 단위로 라이브 미리보기
  - 각 셀 완료 시 즉시 저장
  - 세션 폴더 예) `outputs/2025..._firstPrompt_grid/`
    - `grid.json`: 모델/사이즈/씨드/크랍/그룹명/프롬프트/그룹 변수/셀 메타/참조 썸네일 경로
    - `cells/`: `r{row}_c{col}.jpg`
    - `refs/<group>/01.jpg …` (세션 복원 시 헤더에 표시할 참조 썸네일)
  - **Grid View (load previous session)**: 과거 세션 폴더를 입력하면 헤더(그룹+참조)와 셀 결과를 **재생성 없이** 복원

사용 예시:
- 여러 그룹(남/여, 체형/각도 등) × 여러 프롬프트(복장/배경/연출)를 그리드로 한 번에 비교
- 공통 레퍼런스(브랜드/배경)를 인덱스 1~N에 고정해 두고, 그룹별 얼굴/제품 이미지를 뒤에 붙여 다양한 조합 생성

언제 쓰면 좋은가:
- 대량 비교/선정이 필요할 때(예: 증명사진/프로필/룩북/상품 컷 템플릿 검증)
- “동일 세팅에서 여러 사람” 혹은 “동일 사람에서 여러 연출”을 표로 정리해 의사결정할 때
- 특정 기준(예: 성별/체형/얼굴형 등)에 따라 변수 입력만 바꿔 재사용할 때

4) ### Bulk Runner (폴더 벌크)
- 선택한 폴더 안 **모든 이미지**에 대해 **하나의 공통 프롬프트**로 개별 I2I 실행
- 원본 그대로 전송(전처리 없음), 각 항목은 완료 즉시 저장 및 `metadata.json`에 누적 기록
- 세션 폴더 예) `outputs/2025..._myPrompt_bulk/` 하위에 `<원본파일명>.jpg`로 저장

언제 쓰면 좋은가:
- “같은 프롬프트로 다수의 원본을 한꺼번에 편집”해야 할 때(일괄 보정/복장 교체/배경 통일 등)
- 촬영물 배치 처리가 필요할 때(인물/상품/컷 편집 일괄화)

작동 방식/저장 규칙:
- 입력 폴더의 모든 이미지가 각각 1건의 요청으로 처리
- 결과 파일명은 **입력 원본의 파일명 그대로** `jpg` 확장자로 저장
- 진행 중이라도 항목 단위로 즉시 저장되며 `metadata.json`에 항목이 추가됨

---

### 출력 구조

모든 결과는 `OUTPUT_BASE_DIR`(기본 `outputs`) 아래 세션 폴더로 저장됩니다.

- T2I/I2I: `outputs/<timestamp>_<prompt_snippet>_<suffix>/`
  - `01.jpg, 02.jpg, …`
  - `metadata.json`: `{ prompt, model, sizes[], urls[], created, num_images }`
  - `references/`(I2I): 업로드한 레퍼런스 복사본

- Grid: `outputs/<timestamp>_<first_prompt_snippet>_grid/`
  - `grid.json`, `cells/`, `refs/`
  - `grid.json` 필드 개요(재로드에 사용):
    - `type`: `grid`
    - `created`: 타임스탬프
    - `model`, `width`, `height`, `seed`, `apply_crop`
    - `group_names`: 그룹 헤더 문자열 배열
    - `prompts`: 행 프롬프트 배열
    - `variables`: 그룹별 변수 값 배열(프롬프트 내 `{var}` 치환용)
    - `cells`: `{ "row_col": { file, prompt, size, url } }` 형태로 셀 메타
    - `refs`: `{ "groupIndex": { group, files[] } }` 형태로 헤더에 표시할 참조 썸네일 경로

- Bulk: `outputs/<timestamp>_<prompt_snippet>_bulk/`
  - `<원본파일명>.jpg` (항목별 즉시 저장), `metadata.json`(진행 중 계속 갱신)

---

### 전처리 정책
- **I2I**: 전처리 없음(그대로 전송)
- **Grid 그룹 이미지**: 옵션(기본 ON)일 때만 얼굴 DNN 크랍+1024 리사이즈
- **공통 레퍼런스**: 항상 전처리 없음
- **HEIC/HEIF**: Grid/폴더 로드에서 자동 지원(내부 변환)

권장 해상도/용량:
- I2I/Bulk는 입력 파일이 **10MB 초과 시 거절**되므로 사전 축소 권장(긴 변 2048 내외 JPEG 85~95)
- Grid는 크랍 ON이면 1024×1024로 통일되어 용량 이슈가 적음

---

### API/성능/스트리밍
- 엔드포인트: `POST https://ark.ap-southeast.bytepluses.com/api/v3/images/generations`
- 모델: 기본 `seedream-4-0-250828` (또는 엔드포인트 ID)
- 워터마크: 항상 `false`
- 사이즈: 명시적 `WxH` 또는 `1K/2K/4K`
- 씨드(Seed): 일부 모델에서만 적용됩니다. Seedream 4.x는 현재 씨드를 무시하므로 Grid에서는 입력란이 비활성화됩니다.
- 레퍼런스 제한: 요청당 최대 10장 (Grid에서 공통+그룹 합산 제한 자동 반영)
- 스트리밍: 모델이 지원하면 앱에서 항상 사용되며, 중간 이미지가 표시됩니다. 최종 저장은 완료 시점에 수행됩니다.
- 병렬 실행: Grid/Bulk에 대해 사이드바에서 동시성(스레드) 조절

팁:
- 네트워크 제한/쿼터가 민감하면 동시성 값을 낮게(1~4)
- 프롬프트가 길수록(토큰↑) 응답이 느려질 수 있음 → 짧고 구체적인 표현을 추천

---

### 로그/진행률
- **Live Logs**: 업로드/전처리/페이로드 특성(b64/url/개수)/API 시도&지연/저장/오류 등을 줄 단위로 출력
- **Progress**: Grid/Bulk의 전체 건수 대비 완료 수를 실시간 표시

---

### 문제 해결 가이드
- `ModelNotOpen`: Ark Console에서 모델/엔드포인트 활성화 후 사용
- `InvalidParameter.OversizeImage`: 입력이 10MB 초과(I2I/Bulk는 업로드 전 수동 축소 권장, Grid 크랍 활성화 시 결과는 1024)
- `IndexError` 등 레이아웃 오류: 폴더 로드시 그룹 수가 많거나 비어있는 그룹이 섞인 경우 생길 수 있음 → 상단 숫자 입력은 폴더 로드 시 비활성, 로드된 실제 그룹 수로 자동 처리
- Seedream 4.x 사용 시 Seed 입력이 비활성화되거나 적용되지 않을 수 있습니다(모델에서 무시).

추가 체크리스트:
- Grid에서 공통 레퍼런스 개수가 많을수록(최대 5) 그룹별 레퍼런스 허용량(10−N)이 줄어듭니다
- 스트리밍 사용 시 UI에 중간 이미지가 보이지만, 최종 저장은 완료 시점 파일만 저장됩니다

---

### 커스텀 크로퍼 연결
루트에 `cropper_v2.py`를 두고 `FaceCropper` 클래스를 제공하면 앱이 자동으로 사용합니다.

필수 인터페이스 예시:
- `detect_face_dnn(image)` → `{"box": [x, y, w, h], "confidence": ...}`
- `calculate_balanced_crop_v2(face_center_x, face_center_y, face_size, img_w, img_h, bbox)` → `(x1, y1, x2, y2)`

사용 이유:
- 인물 사진 정사각 크랍을 일정하게 맞추고 그룹별 일관성을 확보하기 위함(증명사진/룩북 등)

세부 동작:
- DNN 검출 성공 시 Balanced Crop v2 로직으로 좌/우/상/하 여유 공간을 고려해 마진을 균형 배분
- bbox가 너무 작거나 경계에 붙으면 중앙 보정
- 실패 시 중앙 정사각 크랍 폴백

변수 프롬프트 예시:
```text
Prompt: ID photo of a korean {gender} wearing a formal grey jacket
Group vars: man_chubby → gender=man, woman_hyunji → gender=woman
```

---

### 보안/키 관리
- `env.local`에 키를 보관(깃 무시됨). 키를 커밋하거나 공유 금지
- 앱은 키를 로그로 출력하지 않음

---

### 개발 정보
- Python 3.9+
- 주요 파일
  - `app.py`: 스트림릿 앱
  - `client.py`: Seedream API 클라이언트(워터마크 false 고정, 스트리밍/재시도/로그 훅)
  - `preprocess.py`: HEIC/크랍 래퍼 등 전처리
  - `utils.py`: 저장/메타데이터 유틸리티

로컬 실행:
```bash
source .venv/bin/activate
streamlit run app.py
```

---

### 자주 묻는 질문(FAQ)

- Q. Grid에서 공통 레퍼런스는 왜 인덱스 1부터 들어가나요?
  - A. 모든 그룹에 동일한 첫 기준 이미지를 보장하기 위함입니다. 공통 3장이라면 각 요청의 레퍼런스 1~3이 동일해 비교가 쉬워집니다.

- Q. Grid 세션을 다시 열면 입력 이미지(참조)도 같이 볼 수 있나요?
  - A. 네. `refs/<group>/xx.jpg`로 저장되어 헤더에 2컬럼으로 표시됩니다.

- Q. Bulk와 Grid의 차이는?
  - A. Bulk는 “하나의 프롬프트 × 다수 원본(개별 저장)”에 최적화, Grid는 “다수 프롬프트 × 다수 그룹(표 비교)”에 최적화입니다.

- Q. HEIC가 안 열릴 때는?
  - A. `pillow-heif` 설치가 필요합니다(이미 `requirements.txt`에 포함). 그래도 실패하면 PNG/JPEG로 변환 후 업로드를 권장합니다.


---

## 워크플로우 예시(End-to-End)

### A) Grid로 증명사진 템플릿 테스트(남/여 그룹 8개 × 프롬프트 3개)
1. 사이드바에서 `Output base folder` 지정, 동시성(Max parallel) 조정
2. Grid 탭에서 프롬프트 3개 입력 (예: 기본/정장/주름보정 90%)
3. 상단 `Common references`에 배경/브랜드 고정 컷 2~3장 업로드(인덱스 1..N)
4. `Load groups from folder`에 `inputs_front3` 지정 → 8개 그룹 자동 로드(하위 폴더 무시)
5. 필요 시 그룹 변수(`{gender}` 등) 입력
6. Balanced Crop v2 체크(기본 ON)
7. Run Grid → 셀별로 스트리밍 미리보기 → 완료 즉시 저장
8. 완료 후 동료와 세션 폴더 공유 → `Grid View (load previous session)`로 재현/리뷰

결과: `outputs/<ts>_<prompt>_grid/`에 `grid.json`, `cells/`, `refs/` 저장. 과거 세션 재로드 시 헤더에 참조 썸네일까지 함께 복원.

### B) Bulk로 촬영본 일괄 보정(공통 프롬프트 1개)
1. Bulk 탭에서 폴더 선택(예: `inputs_shoot`)과 프롬프트 1개 입력
2. Run Bulk → 항목 단위로 생성/저장/메타 갱신
3. 세션 폴더에서 `<원본파일명>.jpg`를 바로 전달

결과: `outputs/<ts>_<prompt>_bulk/`에 즉시 저장. 장시간 작업도 중간 산출물 실시간 확보.

---

## 설정/입력 파일 규칙 정리

- 입력 이미지 포맷: `.jpg/.jpeg/.png/.webp/.heic/.heif` (I2I/Bulk는 10MB 제한)
- EXIF 회전: Grid 업로드/폴더 로드 시 내부적으로 보정, I2I/Bulk는 가능한 사전 정규화 권장
- Grid 레퍼런스 수: 공통 N + 그룹 M ≤ 10 (앱이 자동으로 `M = 10 − N`으로 제한)
- Grid 크랍: 그룹 이미지에만 적용(1024×1024), 공통/ I2I/ Bulk에는 미적용
- 프롬프트 변수 치환: `{var}` → 그룹 탭에서 입력한 값으로 치환

---

## grid.json/metadata.json 스키마 상세

`grid.json` 예시(요약):
```json
{
  "type": "grid",
  "created": "2025-09-12_14-30-12",
  "model": "seedream-4-0-250828",
  "width": 1296,
  "height": 1728,
  "seed": "12345",
  "apply_crop": true,
  "group_names": ["man_chubby", "woman_normal"],
  "prompts": ["ID photo of a korean {gender}", "ID photo ... grey jacket"],
  "variables": [{"gender": "man"}, {"gender": "woman"}],
  "cells": {
    "1_1": {"file": "cells/r1_c1.jpg", "prompt": "ID photo of a korean man", "size": "1296x1728"}
  },
  "refs": {"1": {"group": "man_chubby", "files": ["refs/man_chubby/01.jpg"]}}
}
```

`metadata.json`(Bulk/I2I/T2I) 예시:
```json
{
  "prompt": "...",
  "model": "seedream-4-0-250828",
  "created": "2025-09-12_14-05-51",
  "num_images": 1,
  "items": [
    { "name": "IMG_0001.jpg", "saved": ".../IMG_0001.jpg", "size": "1296x1728", "url": "..." }
  ]
}
```

---

## 문제 해결(추가)

- `ModelNotOpen`: Ark Console에서 모델 활성화 또는 엔드포인트 ID 사용
- `InvalidParameter.OversizeImage`: 10MB 초과 → I2I/Bulk는 사전 축소, Grid 크랍 ON이면 1024로 안전
- `IndexError`/UI 컬럼 오류: 폴더 로드 후 그룹 수가 상단 숫자와 불일치할 때 → 폴더 로드시 상단 숫자 입력은 비활성(앱에서 자동 처리)
- `Connection aborted ('The write operation timed out')`: 동시성 낮추기(1~2), 네트워크 상태 확인, 크기 줄이기
- `Bad 'setIn' index`: 진행률 위젯 초기화 레이스 → 재실행/캐시 초기화

---

## 운영 팁(실전)

- 공통 레퍼런스는 화질이 중요한 로고/브랜드/배경 등에 적합. 품질 저하는 없고 **항상 인덱스 1..N**으로 고정
- Grid 프롬프트 라벨은 컴팩트 폰트/줄간격, 그룹 참조 썸네일은 **2컬럼**으로 꽉 채워 가독성 확보
- HEIC 입력은 내부 변환으로 처리하되, 호환성 이슈가 있을 땐 JPEG로 변환해 업로드 권장
- 로그는 시도/응답시간/저장경로까지 남기므로 실패 재현과 회고에 유용


