import sys
import streamlit as st
import streamlit.components.v1 as components
import sounddevice as sd
import numpy as np
import tempfile
import os
import warnings
from gtts import gTTS                             # TTS

from openai import OpenAI
from scipy.io.wavfile import write
from dotenv import load_dotenv
import faster_whisper                           # STT

# streamlit 서버 임포트
st.set_page_config(layout="centered", initial_sidebar_state="expanded")

# 채팅 UI CSS 스타일 정의
css = """
    <style>
        .chat-container {
            display: flex;
            align-items: flex-start;
            margin-bottom: 20px;
            width: 100%;
        }

        .chat-image {
            width: 100px;
            height: 150px;
            border-radius: 10%;
            margin: 0 15px;
            object-fit: cover;
        }

        .chat-message {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
        }

        .chat-container.ai {
            flex-direction: row-reverse;
            text-align: right;
        }     

        .chat-message.ai {
            margin-right: 0;
        }

        .scroll-buttons {
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            z-index: 1000;
        }
        
        .scroll-buttons button:hover {
            background-color: #f0f0f0;
        }   
    </style>
"""

# JavaScript로 스크롤 기능 추가
js = """
    <script>
        function scrollToTop() {
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
        function scrollToBottom() {
            window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
        }

        var audio = document.querySelector("audio");
        if (audio) {
            audio.autoplay = true;
            audio.muted = false; 
            audio.addEventListener("ended", function() {
                console.log("Audio playback completed.");
            });
        }
    </script>
"""

# HTML로 버튼 추가 (오른쪽 하단에 고정)
html = """
    <div class="scroll-buttons">
        <button onclick="scrollToTop()">⬆️</button>
        <button onclick="scrollToBottom()">⬇️</button>
    </div>
"""

# Streamlit 컴포넌트 생성
components.html(
    css + js + html,
    height=200,  # 필요에 따라 높이 조정
)

st.title("🎙️ SelenaAI ")

sys.stdout.reconfigure(encoding='utf-8')        # 이모티콘 사용 용이
warnings.filterwarnings("ignore")
ai_img = "WOODZ_군복.jpg"
user_img = "사람이미지_1.jpg"

# Whisper 모델 로드 (캐싱 사용)
@st.cache_resource
def load_faster_whisper_model():
    model = faster_whisper.WhisperModel("medium", device="cpu", compute_type="int8")  # 모델 크기 및 compute_type 조정 가능
    return model

model = load_faster_whisper_model()

# 환경변수 로드
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 사이드바 생성
with st.sidebar:
    menu = {"🏠": "home", "🆕": "news", "💬": "history"}
    page = st.radio("Menu", options=menu.keys(), format_func=lambda x: f"{x} {menu[x].capitalize()}")

    if page:
        st.session_state.page = menu[page]

    

# 채팅 기록 유지
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 음성 녹음 함수
def record_audio(duration=6, samplerate=44100):
    st.write("🎤 녹음 중 입니다. 말씀해주세요!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_audio_file.name, samplerate, audio_data)
    return temp_audio_file.name

# 음성 -> 텍스트 : STT (faster-whisper 사용)
def speech_to_text(audio_file):
    segments, info = model.transcribe(audio_file, beam_size=5)
    text = ""
    for segment in segments:
        text += segment.text
    return text

# gTTS 음성 출력 함수
def speak_gtts(text):
    try:
        tts = gTTS(text=text, lang='ko', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            audio_file = open(temp_file.name, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3", start_time=0)
            components.html(
                f"""
                    <script>
                        var audio = document.querySelector("audio");
                        audio.autoplay = true;
                    </script>
                """,
                height=0,
            )
    except Exception as e:
        st.error(f"❌ gTTS 오류 발생: {e}")
    finally:
        if 'audio_file' in locals():
            audio_file.close()

# 최종 TTS 실행 함수
def speak(text):
    speak_gtts(text)

# GPT 응답 생성
def ask_gpt(user_input):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            model="gpt-4-turbo",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

# 음성 녹음 버튼
if st.button("🎤 SelenaAI 입니다. 무엇이든 편하게 질문하세요"):
    audio_file = record_audio()
    st.success("✅ 녹음 완료! 음성 변환 중 입니다...")
    text_input = speech_to_text(audio_file)
    response = ask_gpt(text_input)

    with st.chat_message("user"):
        st.image(user_img, width=50)
        st.write(text_input)

    with st.chat_message("assistant"):
        st.image(ai_img, width=50)
        st.write(response)

    st.session_state.chat_history.append({"role": "user", "content": text_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    speak(response)
    os.remove(audio_file)

st.subheader("💬 대화 기록")
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        if chat["role"] == "user":
            st.image(user_img, width=50)
        else:
            st.image(ai_img, width=50)
        st.write(chat["content"])

if st.button("🛑 종료"):
    st.session_state.chat_history = []
    st.success("대화 기록이 삭제되었습니다.")
