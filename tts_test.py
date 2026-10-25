import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. 모델 로드
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
tts_model = Qwen3TTSModel.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

# 2. 텍스트 및 음성 제어 프롬프트 설정
text = "안녕하세요! 오늘 진행할 인공지능 세미나에 오신 것을 환영합니다."
instruct_prompt = "A warm, energetic male voice speaking clear Korean with an upbeat tone."

# 3. 언어 명칭을 'korean'으로 수정하여 호출
print("음성을 생성 중입니다...")
wavs, sr = tts_model.generate_voice_design(
    text=text,
    instruct=instruct_prompt,
    language="korean" # 'ko' 대신 'korean' 입력
)

# 4. 파일 저장
sf.write("output_qwen3_tts.wav", wavs[0], sr)
print("생성 완료! 'output_qwen3_tts.wav' 파일이 저장되었습니다.")
