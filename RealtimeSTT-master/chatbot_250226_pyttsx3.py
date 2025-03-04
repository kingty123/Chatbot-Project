import sys
sys.stdout.reconfigure(encoding='utf-8')        # 이모티콘 사용 요이
#sys.path.append('thrid_party/Matcha-TTS')       # CosyVoice2-0.5B 사용

import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import tempfile
import os
import openai
import warnings
import pyttsx3
from gtts import gTTS
#import re
from scipy.io.wavfile import write
from dotenv import load_dotenv
#from io import BytesIO


st.set_page_config(layout="centered", initial_sidebar_state="expanded")
# from cosyvoice.cli.cosyvoice import CosyVoice2
# from cosyvoice.utils.file_utils import load_wav
# import torchaudio

warnings.filterwarnings("ignore")
ai_img = "WOODZ_군복.jpg"
user_img = "사람이미지_1.jpg"


# # CosyVoice2 사용 : JIT(최적화된 실행), TensorRT, FP16(반정밀도 연산) 사용 안 함
# cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
# # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# # Zero-shot 음성 합성 (목소리 샘플 기반 음성 생성)
# prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot(
#         '친구가 멀리서 보내준 생일 선물을 받았어. 예상치 못한 깜짝 선물과 따뜻한 축하 메시지가 내 마음을 가득 채웠고, 미소가 꽃처럼 활짝 피어났어.',  
#         '앞으로 나보다 더 잘할 수 있기를 바라!', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # 미세 제어된 음성 합성 (특정 발음 및 효과 포함) : fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248, [laughter] → 문장 중간에 웃음 소리 추가
# for i, j in enumerate(cosyvoice.inference_cross_lingual('그가 그 황당한 이야기를 들려주는 도중, 그는 갑자기 [웃음] 멈추었어. 왜냐하면 스스로도 너무 웃겼기 때문이야! [웃음]', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # Instruct 모드 (스타일 또는 방언 변경) : instruct usage
# for i, j in enumerate(cosyvoice.inference_instruct2(
#         '친구가 멀리서 보내준 생일 선물을 받았어. 예상치 못한 깜짝 선물과 따뜻한 축하 메시지가 내 마음을 가득 채웠고, 미소가 꽃처럼 활짝 피어났어.',  
#         '이 문장을 부산 사투리로 말해줘!', prompt_speech_16k, stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # 스트리밍 기반 TTS : bistream usage, you can use generator as input, this is useful when using text llm model as input
# # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# def text_generator():
#     yield '친구가 멀리서 보내준 생일 선물을 받았어.'
#     yield '예상치 못한 깜짝 선물과 따뜻한 축하 메시지가'
#     yield '내 마음을 가득 채웠고,'
#     yield '미소가 꽃처럼 활짝 피어났어.'

# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '앞으로 나보다 더 잘할 수 있기를 바라!', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)


# Whisper 모델 로드
model = whisper.load_model("base", device="cpu")

# 환경변수 로드
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

st.title("🎙️ SelenaAI ")

with st.sidebar.container():
    # st.header("음성 선택")
    # voice_names = [
    #     "Aria", "Roger", "Sarah", "Laura", "Charlie", "George",
    #     "Callum", "River", "Liam", "Charlotte", "Alice", "Matilda",
    #     "Will", "Jessica", "Eric", "Chris", "Brian", "Daniel", "Lily",
    #     "Bill", "Anna Kim", "Jennie"
    # ]
    # voice_options = {
    #     name: os.environ.get("VOICE_" + name.replace(" ", "_").upper())
    #     for name in voice_names
    # }
    # # 필터링: 등록되지 않은 음성 제거
    # voice_options = {name: vid for name, vid in voice_options.items() if vid}
    # if not voice_options:
    #     st.error("등록된 음성 옵션이 없습니다. .env 파일을 확인해주세요.")
    #     selected_voice_id = None
    # else:
    #     selected_voice = st.selectbox("🔊 음성을 선택하세요", list(voice_options.keys()), key="voice_select") # label_visibility="hidden"
    #     selected_voice_id = voice_options[selected_voice]

    # 속도 조절 슬라이더
    tts_speed = st.slider("음성 속도 조절", min_value=100, max_value=600, value=180, step=5)
    
    # 사용자가 선택할 수 있는 TTS 엔진 목록
    tts_engine = st.selectbox(
        "사용할 TTS 엔진을 선택하세요", 
        ["오프라인", "온라인"])

    # 설정 저장
    st.session_state.tts_engine = tts_engine
    st.session_state.tts_speed = tts_speed



# 채팅 기록 유지
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# 음성 녹음 함수
def record_audio(duration=6, samplerate=44100):
    st.write("🎤 녹음 중 입니다. 말쑴해주세요!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_audio_file.name, samplerate, audio_data)
    return temp_audio_file.name


# 음성 -> 텍스트 : STT
def speech_to_text(audio_file):
    audio = whisper.load_audio(audio_file)
    result = model.transcribe(audio)
    return result["text"]


# 텍스트 -> 음성 : TTS
def speak_pyttsx3(text):
    if not text:
        return  # 빈 텍스트는 변환하지 않음
    
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    # 한국어 음성 선택
    for voice in voices:
        if "Heami" in voice.name or "korean" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()

# gTTS 음성 출력 함수
def speak_gtts(text):
    tts = gTTS(text=text, lang='ko', slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    
    # Streamlit에서 오디오 재생
    st.audio(temp_file.name, format="audio/mp3")
    os.remove(temp_file.name)

# 최종 TTS 실행 함수
def speak(text):
    if tts_engine == "pyttsx3 (오프라인)":
        speak_pyttsx3(text)
    else:
        speak_gtts(text)

# GPT 응답 생성
def ask_gpt(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo", 
            messages=[{"role": "user", "content": user_input}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

# st.set_page_config(layout="centered", initial_sidebar_state="expanded")
# st.title("🎙️ SelenaAI ")


# # 채팅 UI 출력
# for chat in st.session_state.chat_history:
#     with st.chat_message(chat["role"]):
#         if chat["role"] == "user":
#             st.image(user_img, width=50)
#         else:
#             st.image(ai_img, width=50)
#         st.write(chat["content"])


# 음성 녹음 버튼
if st.button("🎤 SelenaAI 입니다. 무엇이든 편하게 질문하세요"):
    audio_file = record_audio()
    st.success("✅ 녹음 완료! 음성 변환 중 입니다...")
    text_input = speech_to_text(audio_file)
    st.write(f"📝 {text_input}")

    response = ask_gpt(text_input)
    st.write(f"🤖 {response}")

    st.session_state.chat_history.append({"role": "user", "content": text_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    speak(response)
    os.remove(audio_file)


st.subheader("💬 대화 기록")
for chat in st.session_state.chat_history:
    # role = "🗣️ 사용자" if chat["role"] == "user" else "🤖 챗봇"
    # st.write(f"**{role}**: {chat['content']}")

    with st.chat_message(chat["role"]):
        if chat["role"] == "user":
            st.image(user_img, width=50)
        else:
            st.image(ai_img, width=50)
        st.write(chat["content"])


# 스크롤 버튼 추가 (버튼을 오른쪽 하단에 고정)
scroll_css = """
    <style>
        .scroll-buttons {
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 1000;
        }
        .scroll-buttons button {
            width: 50px;
            height: 50px;
            font-size: 20px;
            border-radius: 50%;
            border: none;
            background-color: #ffffff;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            cursor: pointer;
        }
        .scroll-buttons button:hover {
            background-color: #f0f0f0;
        }
    </style>
"""
st.markdown(scroll_css, unsafe_allow_html=True)


# JavaScript로 스크롤 기능 추가
scroll_js = """
    <script>
        function scrollToTop() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
        function scrollToBottom() {
            window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
        }
    </script>
"""
st.markdown(scroll_js, unsafe_allow_html=True)

# HTML로 버튼 추가 (오른쪽 하단에 고정)
scroll_buttons_html = """
    <div class="scroll-buttons">
        <button onclick="scrollToTop()">⬆️</button>
        <button onclick="scrollToBottom()">⬇️</button>
    </div>
"""
st.markdown(scroll_buttons_html, unsafe_allow_html=True)

# # 스크롤 버튼 추가
# col1, col2 = st.columns([1, 1])
# with col1:
#     if st.button("⬆️"):
#         st.markdown("<script>window.scrollTo(0, 0);</script>", unsafe_allow_html=True)
# with col2:
#     if st.button("⬇️"):
#         st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)


if st.button("🛑 종료"):
    st.session_state.chat_history = []
    st.success("대화 기록이 삭제되었습니다.")
