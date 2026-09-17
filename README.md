# Commercial DreamBench

본 프로젝트는 상업용 제품 이미지 도메인에서 텍스트-이미지 편집 및 생성(Text-to-Image Editing/Generation) 모델의 객체 보존(Subject Preservation) 및 구성적 이해(Compositional Subject Understanding) 성능을 평가하기 위한 벤치마크 평가 프레임워크입니다.

---

## 📂 디렉토리 구조 및 역할

Git 저장소에 트래킹되는 주요 디렉토리 및 파일 구성과 각 기능은 다음과 같습니다.

### 1. [`assets/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/assets)
평가에 활용되는 데이터셋 통계 및 관련 리소스가 위치합니다. (대용량 이미지 데이터셋 및 평가 루브릭 JSON 파일은 `.gitignore`에 등록되어 로컬에서 관리됩니다.)
*   **[`data/amzn/dataset_stats.html`](file:///root/Desktop/workspace/woosung/commercial-dreambench/assets/data/amzn/dataset_stats.html)**: Amazon 제품 데이터셋 수집 현황 및 메타데이터 통계 시각화 리포트

### 2. [`prompts/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts)
VLM(Visual-Language Model) 및 LLM을 호출하여 루브릭 생성, 정합성 평가, 필터링을 수행할 때 사용하는 프롬프트 템플릿(텍스트 파일) 모음입니다.

*   **데이터 필터링 및 전처리**
    *   [`reference_filtering_instruction.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/reference_filtering_instruction.txt): 레퍼런스 제품 이미지 품질 및 적합성 필터링 지침 프롬프트
*   **루브릭(Rubrics) 생성용**
    *   [`user_prompt_generate_rubric.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_generate_rubric.txt): 기본 제품 식별 속성 기반 평가 루브릭 자동 생성 프롬프트
    *   [`user_prompt_generate_rubric_simple.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_generate_rubric_simple.txt): 간소화된 속성 중심 루브릭 생성 프롬프트
    *   [`user_prompt_generate_rubric_comprehensive.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_generate_rubric_comprehensive.txt): 상세 속성을 포괄하는 종합 루브릭 생성 프롬프트
    *   [`user_prompt_generate_rubric_comprehensive_bbox-titles.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_generate_rubric_comprehensive_bbox-titles.txt): 바운딩 박스(Bounding Box) 타이틀이 포함된 포괄적 루브릭 생성 프롬프트
    *   [`user_prompt_generate_rubric_comprehensive_bbox-titles-coarseNfine.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_generate_rubric_comprehensive_bbox-titles-coarseNfine.txt): Coarse(대분류) 및 Fine(소분류) 계층적 BBox 타이틀을 포함하는 루브릭 생성 프롬프트
*   **루브릭 기반 및 VLM 평가용**
    *   [`user_prompt_evaluate_with_rubric_original.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_evaluate_with_rubric_original.txt): 원본 루브릭 채점 기준 프롬프트
    *   [`user_prompt_evaluate_with_rubric_objective.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_evaluate_with_rubric_objective.txt): 객관식/정량적 판정 루브릭 평가 프롬프트
    *   [`user_prompt_evaluate_with_rubric_considering_generation_context.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_evaluate_with_rubric_considering_generation_context.txt): 편집/생성 맥락(컨텍스트)을 감안한 루브릭 평가 프롬프트
    *   [`user_prompt_evaluate_with_rubric_wo-importance_integer-scale.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_evaluate_with_rubric_wo-importance_integer-scale.txt): 속성별 중요도 가중치를 배제하고 정수 척도로 평가하는 루브릭 프롬프트
    *   [`user_prompt_evaluate_with_crops.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_evaluate_with_crops.txt): 피사체 크롭(Crop) 패치 기반 평가 프롬프트
    *   [`VIEScore_resemblance.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/VIEScore_resemblance.txt): VIEScore 기반 유사성(Resemblance) 측정 프롬프트
    *   [`gpt_prompt_subject_full.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/gpt_prompt_subject_full.txt) / [`user_prompt_subject_full.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/prompts/user_prompt_subject_full.txt): GPT 및 VLM 기반 피사체 전반 보존 평가 프롬프트

### 3. [`scripts/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts)
데이터 왜곡 합성, 이미지 생성(Sampling), 지표 평가(Evaluation), 이미지 유사도 측정 파이프라인을 구동하는 스크립트 모음입니다.

#### 1) 왜곡 생성 파이프라인 ([`scripts/distortion/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/distortion))
평가 지표의 분별력 및 랭킹 정합성을 검증하기 위해 레퍼런스 이미지에 제어된 왜곡을 주입하는 모듈입니다.
*   [`noise_denoise_one_step.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/distortion/noise_denoise_one_step.py): Stable Diffusion XL과 DDIMScheduler를 이용한 단일 스텝 노이즈-디노이즈(Noise-Denoise / FlowFixer 스타일) 왜곡 파이프라인
*   [`noise_denoise_amzn_variation_masked_batch.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/distortion/noise_denoise_amzn_variation_masked_batch.py): 마스크 기반으로 아마존 제품 변형 이미지에 배치 단위 왜곡을 주입하는 스크립트
*   [`lib/mask_dino_adaptive.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/distortion/lib/mask_dino_adaptive.py): DINO 특징 맵 기반 적응형 세그멘테이션 마스크 생성 라이브러리
*   *(참고: 의존성 패키지는 루트의 [`requirements_flowfixer.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_flowfixer.txt)를 참조합니다.)*

#### 2) 모델 샘플링 ([`scripts/sampling/amzn/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn))
다양한 생성/편집 모델을 통해 아마존 제품 레퍼런스로부터 변형 이미지를 생성하는 스크립트입니다.
*   **기본 모델 생성 스크립트**
    *   [`flux_kontext_generation.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/flux_kontext_generation.py): FLUX-Kontext 파이프라인 기반 이미지 생성
    *   [`flux-klein.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/flux-klein.py): FLUX-Klein 모델 기반 샘플 생성
    *   [`QWEN-Image-Edit_generation.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/QWEN-Image-Edit_generation.py): Qwen-Image-Edit 기반 편집 샘플 생성
    *   [`qwen-image.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/qwen-image.py): Qwen 이미지 생성 스크립트
    *   [`diptych_generation.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/diptych_generation.py): Diptych 프롬프팅 기법 기반 생성 스크립트
    *   [`sample_all_beauty_flux.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/sample_all_beauty_flux.py): Beauty 카테고리 전체에 대한 FLUX 일괄 생성 스크립트
*   **왜곡 주입 데이터 기반 생성 스크립트 (Distorted-SDG)**
    *   [`sample_distorted-SDG_flux-kontext.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/sample_distorted-SDG_flux-kontext.py): 왜곡 단계별 레퍼런스에 대한 FLUX-Kontext 샘플링
    *   [`sample_distorted-SDG_flux-klein.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/sample_distorted-SDG_flux-klein.py): 왜곡 단계별 레퍼런스에 대한 FLUX-Klein 샘플링
    *   [`sample_distorted-SDG_qwen-image-edit.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/sampling/amzn/sample_distorted-SDG_qwen-image-edit.py): 왜곡 단계별 레퍼런스에 대한 Qwen-Image-Edit 샘플링

#### 3) 평가 파이프라인 ([`scripts/evaluation/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation))
다양한 평가 메트릭을 산출하고 모델 성능 및 지표 간 정합성을 검증하는 모듈입니다.
*   **루브릭 생성**
    *   [`generate_rubrics.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/generate_rubrics.py): 로컬 VLM을 사용하여 레퍼런스 이미지 기반 루브릭 JSON 자동 생성
    *   [`generate_rubrics_efficient.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/generate_rubrics_efficient.py): 배치 및 효율적 파이프라인이 적용된 루브릭 생성 스크립트
*   **VLM 및 벤치마크 평가**
    *   [`evaluate_compositional.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/evaluate_compositional.py) / [`evaluate_compositional_efficient.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/evaluate_compositional_efficient.py): 제품의 구성적 요소(Compositional attributes)에 대한 심층 평가
    *   [`evaluate_bbox-crop_efficient.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/evaluate_bbox-crop_efficient.py): Bounding Box 패치 크롭 기반 고해상도 세부 속성 평가
    *   [`eval_dreambench_plus.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/eval_dreambench_plus.py) / [`eval_dreambench_plus_efficient.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/eval_dreambench_plus_efficient.py): DreamBench++ 기준 로컬 VLM 평가
    *   [`eval_viescore.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/eval_viescore.py) / [`eval_viescore_efficient.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/eval_viescore_efficient.py): VIEScore 프레임워크 기반 유사도 평가
    *   [`eval_qwen-reranker.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/eval_qwen-reranker.py): Qwen3-VL-Reranker-2B를 활용한 레퍼런스-생성물 간 정합도 랭킹 평가
    *   [`eval_clip.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/eval_clip.py): 생성 이미지 세트 간 CLIP 유사도 일괄 측정
    *   [`eval_dino.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/eval_dino.py): 생성 이미지 세트 간 DINO 유사도 일괄 측정
    *   [`eval_consistency.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/eval_consistency.py): 생성 결과물 간 일관성(Consistency) 측정
*   **통계 분석 도구**
    *   [`utils/calculate_rank_alignment.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/evaluation/utils/calculate_rank_alignment.py): 평가 지표별 Kendall's tau, Spearman rho, Pearson r 및 일치율(Concordance Rate)을 계산하여 랭킹 정합성을 분석하는 도구

#### 4) 이미지 유사도 ([`scripts/image-similarity/`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/image-similarity))
*   [`CLIP.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/image-similarity/CLIP.py): CLIP 임베딩 추출 및 코사인 유사도 계산
*   [`DINO.py`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/image-similarity/DINO.py): DINOv2 임베딩 추출 및 유사도 계산
*   [`README.md`](file:///root/Desktop/workspace/woosung/commercial-dreambench/scripts/image-similarity/README.md): 임베딩 평가 환경 가이드

### 4. 환경별 의존성 요구사항 파일 (`requirements_*.txt`)
각 기능별로 최적화된 Python 패키지 의존성 파일 목록입니다.
*   [`requirements_build-dataset.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_build-dataset.txt): Amazon 제품 리뷰 데이터셋 다운로드 및 전처리 모듈용 패키지
*   [`requirements_amzn-filtering-ui.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_amzn-filtering-ui.txt): 데이터셋 필터링 및 검수 웹 UI(Flask 등) 구동용 패키지
*   [`requirements_flowfixer.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_flowfixer.txt): FlowFixer / SDXL 노이즈-디노이즈 왜곡 생성 및 DINO 적응형 마스크용 패키지
*   [`requirements_sample.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_sample.txt): 이미지 생성 모델(Diffusers, FLUX-Kontext, FLUX-Klein, Qwen 등) 샘플링 패키지
*   [`requirements_eval.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_eval.txt): VLM 추론, 루브릭 자동 생성 및 DreamBench++/구성성 평가용 패키지
*   [`requirements_qwen-rereanker.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_qwen-rereanker.txt): Qwen3-VL-Reranker-2B 기반 이미지 랭킹 평가용 패키지
*   [`requirements_vllm.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_vllm.txt): vLLM 기반 VLM/LLM 고속 대량 추론 환경용 패키지
*   [`requirements_clip-dino.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_clip-dino.txt): CLIP 및 DINOv2 백본 임베딩 유사도 측정 패키지

---

## ⚙️ 개발 및 실행 환경 (가상환경)

본 프로젝트는 모델 및 라이브러리 간 CUDA/Torch 및 패키지 충돌을 방지하기 위해 작업 영역별로 분리된 가상환경(Virtual Environment)을 운용합니다.

| 가상환경 이름 | Python 버전 | 해당 요구사항 파일 | 주요 용도 및 설명 |
| :--- | :---: | :--- | :--- |
| **`.venv-build-dataset`** | **Python 3.12** (3.12.13) | [`requirements_build-dataset.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_build-dataset.txt) | Amazon 리뷰 데이터셋 수집 및 전처리 |
| **`.venv_visualizer`** | **Python 3.8 ~ 3.12** (3.8.10) | [`requirements_amzn-filtering-ui.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_amzn-filtering-ui.txt) | 데이터셋 필터링 웹 UI 구동 |
| **`.venv-flowfixer`** | **Python 3.8** (3.8.10) | [`requirements_flowfixer.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_flowfixer.txt) | FlowFixer / SDXL 1-step 왜곡 합성 및 마스킹 |
| **`.venv-FLUX-Kontext`** | **Python 3.12** (3.12.13) | [`requirements_sample.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_sample.txt) | FLUX-Kontext, FLUX-Klein, Qwen-Image-Edit 등 샘플링 |
| **`.venv-niihau-11112`** | **Python 3.12** (3.12.13) | [`requirements_eval.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_eval.txt) | Qwen-VL 기반 루브릭 생성, DreamBench++, VIEScore 등 평가 |
| **`.venv-qwen-reranker`**| **Python 3.10** (3.10.20) | [`requirements_qwen-rereanker.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_qwen-rereanker.txt) | Qwen3-VL-Reranker-2B 유사도 순위 평가 |
| **`.venv-vllm`** | **Python 3.11** (3.11.15) | [`requirements_vllm.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_vllm.txt) | vLLM 기반 초고속 대량 추론 파이프라인 |
| *(CLIP/DINO 환경)* | **Python 3.11+** | [`requirements_clip-dino.txt`](file:///root/Desktop/workspace/woosung/commercial-dreambench/requirements_clip-dino.txt) | CLIP 및 DINOv2 기반 임베딩 코사인 유사도 평가 (`TypeAlias` 지원 필요) |

---

## 🖥️ 로컬 시스템 사양 및 인프라 구성 (System Specifications)

현재 구동 및 벤치마크 평가를 수행하는 로컬 서버의 핵심 하드웨어 및 드라이버 환경 사양입니다.

*   **GPU (Graphics Processing Unit)**:
    *   **모델**: **2 × NVIDIA A100 80GB PCIe**
    *   **총 GPU 장치 수**: 2개
    *   **VRAM**: 개당 80GB (총 160GB VRAM 지원)
*   **NVIDIA 드라이버 버전 (Driver Version)**: `580.173.02`
*   **드라이버 지원 CUDA 버전 (Supported CUDA)**: `CUDA 13.0`
*   **호스트 시스템 CUDA 컴파일러 (`nvcc`)**: `CUDA 11.2` (V11.2.152)
    *   *(참고: 각 가상환경의 PyTorch 패키지는 자체 내장된 CUDA 12.1 ~ 12.4 런타임 라이브러리를 바인딩하여 구동됩니다.)*
*   **운영체제 (OS)**: Ubuntu 20.04.2 LTS (Focal Fossa)
*   **리눅스 커널 (Kernel)**: `5.15.0-186-generic`
*   **CPU**: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
