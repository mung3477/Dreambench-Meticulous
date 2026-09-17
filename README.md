# Commercial DreamBench

본 프로젝트는 상업용 제품 이미지 도메인에서 텍스트-이미지 편집 및 생성(Text-to-Image Editing/Generation) 모델의 객체 보존(Subject Preservation) 및 구성적 이해(Compositional Subject Understanding) 성능을 평가하기 위한 벤치마크 평가 프레임워크입니다.

---

## 📂 디렉토리 구조 및 역할

프로젝트 루트 디렉토리의 전체 구조와 각 폴더/파일의 기능은 다음과 같습니다.

### 1. [`assets/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/assets)
평가에 활용되는 데이터셋 및 모델 평가 기준(Rubrics)이 저장되는 디렉토리입니다.
*   **[`data/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/assets/data)**: 평가용 소스 데이터가 위치합니다.
    *   **[`amzn/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn)**: 아마존 제품 리뷰 데이터셋 기반의 이미지 및 메타데이터 파일들이 카테고리(`raw_meta_*`)별로 구성되어 있습니다.
    *   **[`commercial-dreambench_PoC/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/assets/data/commercial-dreambench_PoC)**: 개념 검증(PoC) 단계에서 사용된 데이터셋입니다.
*   **[`rubrics/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/assets/rubrics)**: VLM(Visual-Language Model)이 레퍼런스 이미지를 바탕으로 자동 생성한 평가 루브릭 파일들이 해상도별(`512x512`, `1024x1024`)로 관리됩니다.

### 2. [`prompts/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts)
VLM(예: Qwen-VL 등)을 호출할 때 사용하는 프롬프트 템플릿(텍스트 파일)들이 저장되어 있습니다.
#### 데이터 preprocessing용
*   [`reference_filtering_instruction.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/reference_filtering_instruction.txt): 레퍼런스 제품 이미지 필터링을 위한 프롬프트 지침
*   [`variation_filtering_instruction.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/variation_filtering_instruction.txt): 베리에이션(변형) 이미지 필터링을 위한 프롬프트 지침


#### 평가용
*   [`user_prompt_generate_rubric.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_generate_rubric.txt): 레퍼런스 이미지를 분석하여 제품 고유의 식별 속성(Brand, Variant, 구조 등)별 평가 루브릭을 추출하기 위한 프롬프트
*   [`user_prompt_evaluate_with_rubric.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_evaluate_with_rubric.txt): 생성된 이미지 샘플과 생성 대상 루브릭을 매칭하여 객체 보존 상태를 평가하기 위한 프롬프트

### 3. [`rating/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/rating)
평가 파이프라인의 실행 결과(점수 및 판정 결과)와 중간 결과물들이 저장되는 디렉토리입니다.

### 4. [`samples/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/samples)
각 모델을 사용하여 생성 및 편집한 결과 이미지(샘플)들이 저장되는 공간입니다. 수집된 모든 샘플은 **Diptych Prompting**, **Flux-Kontext**, **Qwen-Image-Edit** 모델을 통해 생성되었습니다.
*   **[`amzn/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/samples/amzn)**: 아마존 데이터셋(`assets/data/amzn`)을 소스로 하여 생성된 샘플 결과물입니다.
*   **[`commercial-PoC/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/samples/data)**: PoC 데이터셋(`assets/data/commercial-PoC`)을 기반으로 생성된 이미지 샘플 결과물입니다.
    *   **`commercial-PoC/noised-*/` (예: `noised-0.25x`, `noised-0.5x` 등)**: PoC 데이터셋에 대해 FlowFixer 1-step noise-denoise 기법을 적용한 결과물입니다. 폴더명 끝의 배수(`0.25x`, `0.5x` 등)는 noise-denoise 수행 이전의 이미지 스케일(크기)을 나타내며, 배수가 작아질수록 원본 정보 손실 및 왜곡이 심해집니다.
    *   **`commercial-PoC/prompt_*/` (예: `prompt_naive_ver`, `prompt_sdg_ver_omitting_text` 등)**: PoC 데이터셋에 대해 다양한 프롬프트 추출/정제 버전에 대응하여 샘플링한 결과 이미지 세트입니다.

### 5. [`scripts/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts)
데이터 통합, 프롬프트 추출, 샘플 생성 및 평가의 모든 파이프라인을 자동화하고 지원하는 스크립트 모음입니다.
*   [`generate_rubrics.sh`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/generate_rubrics.sh): 평가용 루브릭 생성을 백그라운드에서 구동하는 쉘 스크립트입니다.
*   [`eval_db++.sh`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/eval_db++.sh) / [`eval_compositional.sh`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/eval_compositional.sh): DreamBench++ 기준 평가 및 구성적 평가(Compositional Evaluation)를 실행하기 위한 통합 쉘 스크립트입니다.
*   **[`dataset-generation/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/dataset-generation)**: 소스 데이터로부터 프롬프트를 추출하고 데이터셋을 구축/정제하는 모듈입니다.
    *   **[`amzn/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/dataset-generation/amzn)**: 아마존 데이터셋 대상 구축 및 필터링/시각화 스크립트가 포함되어 있습니다.
        *   `build_dataset.py`: `Amaon-review-2023`으로부터 평가 데이터셋을 자동으로 구축하는 스크립트
        *   `clean_samples.py`: `visualize.py`를 이용한 데이터셋 필터링 이후 필요 없는 sample을 지우는 스크립트.
        *   [`visualize.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/dataset-generation/amzn/visualize.py): 웹 기반 데이터 필터링 UI 스크립트
        *   [`visualize_variants.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/dataset-generation/amzn/visualize_variants.py): 레퍼런스 이미지, 생성 이미지 베리에이션, Ground Truth 이미지 및 프롬프트 텍스트를 한눈에 비교할 수 있는 시각화용 웹 UI 서버 스크립트 (CLI `--model_dir` 인자를 통해 특정 모델의 결과 디렉토리를 로드할 수 있습니다)
    *   **[`commercial-PoC/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/dataset-generation/commercial-PoC)**: PoC 데이터셋 관련 프롬프트 생성 스크립트가 포함되어 있습니다. (`generate_prompts.py`, `generate_sdg_prompts.py`)
*   **[`evaluation/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation)**: 평가 파이프라인 관련 모듈입니다.
    *   `generate_rubrics.py`: 로컬 VLM을 사용하여 레퍼런스 이미지 기반 루브릭 JSON을 자동 생성합니다.
    *   `evaluate_with_local_vlm.py`: 로컬 VLM을 통해 DreamBench++ 평가 파이프라인을 실행합니다.
    *   `evaluate_compositional.py`: 로컬 VLM을 통해 reasoning 기반 Ours 평가 파이프라인을 실행합니다.
    *   `visualize_scores.py`: 평가 완료 후 점수를 집계하고 시각화 차트 및 HTML 보고서 파일로 렌더링합니다.
*   **[`image-similarity/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/image-similarity)**: 이미지 특징 분석 및 단일 이미지 쌍 평가 모듈입니다.
    *   `CLIP.py` / `DINO.py`: CLIP 및 DINO 기반의 특징 벡터 추출 및 유사도 계산을 수행합니다.
    *   [`dreambench++.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/image-similarity/dreambench++.py): 단일 레퍼런스-대상 이미지 쌍을 입력받아 `category="subject"`, `few_shot=0` 등의 조건으로 local VLM 기반 DreamBench++ 평가 점수를 측정하는 독립 평가 스크립트입니다.
    *   [`dreambench++.ipynb`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/image-similarity/dreambench++.ipynb): 모델을 GPU 메모리에 상주시킨 채로 다양한 이미지 쌍에 대해 빠르게 반복 평가할 수 있도록 구성된 Jupyter Notebook 버전입니다.
*   **[`sampling/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling)**: 이미지 생성/편집 샘플링 모듈입니다.
    *   **[`amzn/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn)**: 아마존 데이터셋에 대한 이미지 생성/편집 스크립트입니다. (`flux_kontext_generation.py`, `QWEN-Image-Edit_generation.py`, `diptych_generation.py`, `sample_amzn.sh`)
    *   **[`commercial-PoC/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/commercial-PoC)**: PoC 데이터셋에 대한 이미지 생성/편집 스크립트입니다. (`flux_kontext_generation.py`, `QWEN-Image-Edit_generation.py`, `diptych_generation.py`, `sample.sh`)
*   **[`utils/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/utils)**: 공통 유틸리티 스크립트 모듈입니다.
    *   [`integrate_assets.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/utils/integrate_assets.py): `assets/data`의 원본 메타데이터를 기반으로 생성된 샘플 디렉토리에 레퍼런스 이미지와 Ground Truth 베리에이션 이미지, 프롬프트 텍스트를 구조에 맞게 매핑 및 정리해주는 통합 스크립트입니다.

### 6. 기타 루트 파일
*   [`compositional_reasoning_testbed.ipynb`](file:///root/Desktop/workspace/woosung/commercial-dreambench/compositional_reasoning_testbed.ipynb): 제품 이미지 구성성 평가 또는 프롬프트 처리를 위한 대화형 주피터 노트북 테스트베드입니다.
*   **`requirements_*.txt`**: 환경별 Python 라이브러리 설치 요구사항 목록입니다.
    *   [`requirements_amzn-filtering-ui.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_amzn-filtering-ui.txt): 데이터 필터링 UI 실행 관련 의존성 패키지
    *   [`requirements_eval.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_eval.txt): 평가(Evaluation)용 의존성 패키지
    *   [`requirements_sample.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_sample.txt): 샘플 생성(Sampling)용 의존성 패키지

---

## ⚙️ 개발 및 실행 환경 (가상환경)

본 프로젝트는 독립된 가상환경(Virtual Environment)을 활용하여 구동됩니다.
*   **FLUX-Kontext / 샘플 생성 환경**:
    *   가상환경 경로: `.venv-FLUX-Kontext`
    *   활성화 명령: `source .venv-FLUX-Kontext/bin/activate`
*   **VLM / DreamBench++ 평가 및 루브릭 생성 환경**:
    *   가상환경 경로: `.venv-niihau-11112` 또는 `../dreambench_plus/.venv-niihau11112`
    *   활성화 명령: `source .venv-niihau-11112/bin/activate`
