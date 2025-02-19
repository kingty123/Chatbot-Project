import whisper
import sounddevice as sd
import numpy as np
import pyttsx3
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import pyautogui
import time

# Whisper 모델 로드
model = whisper.load_model("base", device="cpu")

# 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = API_KEY

# Langchain 설정
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
cache_dir = LocalFileStore("./.cache/")
splitter = CharacterTextSplitter.from_tiktoken_encoder(separator="\n", chunk_size=600, chunk_overlap=100)
retriever = FAISS.load_local("./refer.index", OpenAIEmbeddings())

# Whisper를 통한 STT (음성 -> 텍스트 변환)
def recognize_speech():
    samplerate = 16000
    duration = 5  # 초기 설정 값 (5초)
    
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    
    # Whisper 모델을 사용하여 음성을 텍스트로 변환
    audio_input = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_input, n_mels=model.dims.n_mels).to(model.device)
    result = model.transcribe(mel)
    return result['text']


# 음성 감지 종료 시점 (사용자가 말하는 마지막 시점까지)
def listen_until_silence(silence_threshold=0.02, max_duration=10):
    """
    사용자 음성을 듣고, 말하는 중이 아니면 녹음을 종료하는 함수.
    silence_threshold: 음성이 감지되지 않은 경우로 판단하는 임계값 (0.02는 기준값으로 변경 가능)
    max_duration: 최대 녹음 시간 (초)
    """
    samplerate = 16000
    silence_duration = 0
    audio_data = []
    
    with sd.InputStream(callback=lambda indata, frames, time, status: audio_data.append(indata), channels=1, samplerate=samplerate, dtype='int16'):
        print("Listening for your speech...")
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_duration:
                break  # 최대 녹음 시간 초과 시 종료

            # 실시간 오디오 신호의 평균 에너지 (음성 신호의 강도)
            energy = np.linalg.norm(audio_data[-1]) ** 2
            if energy < silence_threshold:  # 음성이 멈추면 종료
                silence_duration += 1
                if silence_duration > 2:  # 2초 이상 소리가 없으면 멈춘 것으로 판단
                    print("No speech detected for a while. Stopping...")
                    break
            else:
                silence_duration = 0  # 음성이 감지되면 silence_duration 리셋
                
    # 받은 음성 데이터를 Whisper로 변환
    audio = np.concatenate(audio_data, axis=0)
    audio_input = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_input, n_mels=model.dims.n_mels).to(model.device)
    result = model.transcribe(mel)
    
    return result['text']


# GPT-4를 사용하여 질문에 대한 답변을 생성하는 함수
def ask_gpt(user_question):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant."), ("user", user_question)]
    )
    
    result = llm.invoke({"question": user_question, "context": retriever})
    return result.content


# TTS (텍스트 -> 음성 변환)
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# 음성 명령 감지
def detect_keyword(audio):
    audio_input = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_input, n_mels=model.dims.n_mels).to(model.device)
    result = model.transcribe(mel)
    if "selena" in result['text'].lower():  # "selena" 감지
        return True
    return False


# 음성을 처리하고 명령을 처리하는 함수
def process_audio_command():
    # 사용자 음성 인식 후 질문 받기
    print("음성을 기다리는 중...")
    audio = listen_until_silence()
    
    # 음성이 "selena"를 포함하는지 확인
    if detect_keyword(audio):
        print("Selena 활성화됨!")
        # 사용자가 질문을 할 때까지 대기
        question = listen_until_silence()
        print(f"사용자 질문: {question}")
        
        # GPT-4로 답변을 생성
        answer = ask_gpt(question)
        print(f"챗봇 답변: {answer}")
        
        # 음성으로 답변 전달
        speak(answer)

# Streamlit UI
st.set_page_config(layout="wide", page_title="AI Assistant")
st.title("AI Assistant")

# 질문 입력과 답변
question = st.text_input("Ask something", value=st.session_state.get("question", ""))

if question and st.button('Submit'):
    st.session_state["question"] = question
    answer = ask_gpt(question)
    st.write(answer)
    speak(answer)  # 음성으로 답변

# 동기식으로 실행
if __name__ == "__main__":
    # 음성 처리 및 질문-답변 과정을 동기적으로 실행
    process_audio_command()
