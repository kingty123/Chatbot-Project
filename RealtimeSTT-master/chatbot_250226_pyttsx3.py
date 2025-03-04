import sys
sys.stdout.reconfigure(encoding='utf-8')        # 이모티콘 사용 용이

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
from scipy.io.wavfile import write
from dotenv import load_dotenv


st.set_page_config(layout="centered", initial_sidebar_state="expanded")

warnings.filterwarnings("ignore")
ai_img = "IMAGE-you-want"
user_img = "사람이미지_1.jpg"


# Whisper 모델 로드
model = whisper.load_model("base", device="cpu")

# 환경변수 로드
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

st.title("🎙️ SelenaAI ")

with st.sidebar.container():

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
    st.write("🎤 녹음 중 입니다. 말씀씀해주세요!")
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



if st.button("🛑 종료"):
    st.session_state.chat_history = []
    st.success("대화 기록이 삭제되었습니다.")
