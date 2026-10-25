# Qwen3-TTS 완전 가이드: 설치부터 실행 및 트러블슈팅

본 문서는 **Qwen3-TTS (1.7B VoiceDesign)** 모델을 Linux (Ubuntu / Conda) 환경에서 처음부터 구축하고 구동하기 위한 종합 매뉴얼입니다.

---

## 1. 사전 필수 요구사항 (System Dependencies)

Qwen3-TTS 모델의 음성 토크나이저 및 라이브러리 구동을 위해서는 시스템 레벨의 오디오 패키지(**SoX**) 설치가 필수입니다.

```bash
# Ubuntu/Debian 기반 시스템 오디오 라이브러리 설치
sudo apt update && sudo apt install -y sox libsox-fmt-all
```

---

## 2. Python 환경 및 PyTorch/Cuda 설정

NVIDIA GPU (RTX 30/40/50 시리즈 등)를 지원하는 PyTorch 패키지를 설치합니다. Conda 가상환경 내에 직접 패키지를 설치해야 경로 인식 문제가 발생하지 않습니다.

```bash
# 1. (선택사항) Qwen3-TTS 전용 Conda 가상환경 생성 및 활성화
conda create -n qwen-tts python=3.10 -y
conda activate qwen-tts

# 2. CUDA 12.8 대응 PyTorch / TorchAudio 설치
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Qwen3-TTS 및 필수 패키지 설치
python -m pip install qwen-tts soundfile
```

### PyTorch 및 GPU 가속 설치 검증
```bash
python -c "import torch, torchaudio; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
```
*(결과에 `True`가 출력이 되어야 합니다.)*

---

## 3. Qwen3-TTS 사용 코드 (`tts_test.py`)

`Qwen3-TTS-12Hz-1.7B-VoiceDesign` 모델은 자연어 지시문(`instruct`)을 통해 성별, 연령, 톤앤매너, 감정 등을 완벽하게 조정할 수 있습니다.

> **주의사항:** 
> - `generate` 대신 **`generate_voice_design`** 메서드를 사용해야 합니다.
> - 언어 매개변수는 축약어(`'ko'`)가 아닌 전체 영문 표기(`'korean'`)를 사용해야 합니다.

### 실행 스크립트 작성
```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. 모델 로드 (bfloat16 사용)
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
tts_model = Qwen3TTSModel.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

# 2. 발화 텍스트 설정
text = "안녕하세요! 오늘 진행할 인공지능 세미나에 오신 것을 환영합니다."

# 3. 음색/성별/톤 프롬프트 설정 (원하는 스타일 선택)
# [여성 - 차분하고 전문적인 톤]
instruct_prompt = "A calm, intelligent female voice speaking clear and polite Korean with a professional tone."

# [여성 - 밝고 친근한 톤]
# instruct_prompt = "A bright, friendly young female voice speaking natural and cheerful Korean."

# [남성 - 웅장하고 깊은 내레이터 톤]
# instruct_prompt = "A deep, powerful, and serious male voice speaking slow Korean like a documentary narrator."

# 4. 음성 생성
print("음성을 생성 중입니다...")
wavs, sr = tts_model.generate_voice_design(
    text=text,
    instruct=instruct_prompt,
    language="korean"  # 'ko' 대신 반드시 'korean' 입력
)

# 5. 오디오 파일 저장
output_path = "output_qwen3_tts.wav"
sf.write(output_path, wavs[0], sr)
print(f"생성 완료! '{output_path}' 파일이 저장되었습니다.")
```

### 실행
```bash
python tts_test.py
```

---

## 4. 트러블슈팅 가이드

| 오류 현상 / 메시지 | 원인 | 해결 방법 |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'torch'` | Conda 파이썬과 `pip` 설치 경로 불일치 | `python -m pip install ...` 명령을 사용해 현재 실행되는 파이썬 커널에 직접 패키지를 설치합니다. |
| `SoX could not be found!` / `_torchaudio.abi3.so: undefined symbol` | 시스템 SoX 누락 및 Torch-TorchAudio ABI 버전을 새로 맞춤 | `sudo apt install sox libsox-fmt-all` 실행 후 `torch` 및 `torchaudio` 재설치 |
| `AttributeError: 'Qwen3TTSModel' object has no attribute 'generate'` | 버전별 메서드명 불일치 | `generate()` 대신 `generate_voice_design()` 호출 |
| `ValueError: Unsupported languages: ['ko']` | 언어 코드 미지원 | `language="ko"` 대신 `language="korean"`으로 전체 문자열 입력 |
| `RepositoryNotFoundError: ...-VoiceClone` | 독립적인 VoiceClone 레포지토리 없음 | VoiceDesign 모델에서 제공하는 자연어 `instruct` 기능을 사용하여 음색 제어 |

---

## 5. 추론 속도 최적화 (선택 사항)

NVIDIA RTX 시리즈 GPU에서 더 빠른 추론 속도를 얻기 위해 `flash-attn` 라이브러리를 설치할 수 있습니다.

```bash
pip install flash-attn --no-build-isolation
```