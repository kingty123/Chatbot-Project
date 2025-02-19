# Whisper 사용
import whisper
import sounddevice as sd
import numpy as np
#import time
import threading                    # 비동거 작업이나, 동시에 여러 작업 처리
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import base64
import pyttsx3

from langchain.chat_models import ChatOpenAI
#from langchain_core.pydantic_v1 import BaseModel
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from PIL import Image
from streamlit_geolocation import streamlit_geolocation



model = whisper.load_model("base", device="cpu")

from langchain_openai import OpenAIEmbeddings

# 환경 변수를 로드합니다.
load_dotenv()

# API 키를 환경 변수에서 가져옵니다.
API_KEY = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = API_KEY 


# Langchain을 활용하기 위한 설정과 RAG 설정을 진행합니다.
llm = ChatOpenAI(model='gpt-4',
    temperature=0.7)
cache_dir = LocalFileStore("./.cache/practice/")
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)
#retriever = FAISS.load_local("./refer.txt", OpenAIEmbeddings(), allow_dangerous_deserialization=True)


loader = UnstructuredFileLoader("refer.txt")
docs = loader.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
vectorstore = FAISS.from_documents(docs, cached_embeddings)
retriever = vectorstore.as_retriever()



#from streamlit_geolocation import streamlit_geolocation


# Streamlit 세션 상태를 초기화합니다. 이는 대화 내역을 저장하는 데 사용됩니다.
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'mode' not in st.session_state:
    st.session_state.mode = "voice" # 기본 모드 = 음성

# 이미지 경로 설정
ai_avatar = "karina_1-removebg-preview.png"  # AI 아바타 이미지
user_avatar = "사람이미지_1.jpg"  # 사용자 아바타 이미지

# 이미지 Base64 변환 함수
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()



# 사용자 질문에 대한 응답을 처리하는 함수입니다.
def ask_gpt(user_question):
    # 이전 대화 내역을 기반으로 CHATGPT에게 요청할 쿼리를 생성합니다.
    conservation_history = "\n".join(st.session_state.chat_history[-50:])

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            당신은 'Selena'라는 차량용 인공지능 비서입니다. 친근하고 유머러스하게 사용자의 질문에 답변하며, 운전 중에도 안전하고 유용한 정보를 제공하는 것이 주된 역할입니다. 사용자가 질문을 하면 정확하고 명확한 답변을 주되, 운전 중에도 편안하게 이해할 수 있도록 간결하고 친근하게 응답해주세요. 아래 지침을 따라주세요:

            1. Selena or 셀레나 라고 부르는 음성에 반응합니다.
            2. 사용자의 질문을 신중하게 분석합니다.
            3. 친절하고 명확하고 간결하며, 운전 중에도 쉽게 이해할 수 있도록 답변합니다.
            4. 질문이 복잡하면 이해하기 쉬운 작은 부분으로 나누어 설명합니다.
            5. 예시나 비유를 사용하여 개념을 쉽게 풀어 설명합니다.
            6. 답변을 잘 모르겠다면 그 사실을 인정하고, 관련된 정보를 찾을 수 있는 방법을 제시합니다.
            7. 사용자가 더 알아볼 수 있도록 후속 질문을 유도하거나 관련된 주제를 제안합니다.
            8. 항상 긍정적이고 격려하는 태도를 유지합니다.
            9. 운전 중에는 안전을 최우선으로 고려하여 대화를 유도합니다.
            \n\n
            {context}",
            """
        ),
        ("human", "{question}")
    ]
    )

    # LLM 체인 실행행
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    result = chain.invoke(user_question)
        
    return result.content





# 질문 입력 클리어 함수
def clear_input():
    st.session_state.question = ""


# 음성을 합성하는 함수 : TTS
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

from RealtimeSTT import AudioToTextRecorder
import pyautogui




# 키워드 감지 함수(기본 예시)
def detect_keyword(audio):
    #return np.random.rand() > 0.97  # 3% 확률로 키워드 감지
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    result = model.transcribe(mel)

    # 감지된 텍스트에서 "Hey"를 찾음
    if "Selena" in result['text'].lower():
        return True
    return False




# process_command() : 특정 키워드를 감지했을 때 호출되며, 사용자의 음성 명령 처리
def process_command(text):
    print(f"사용자 요구사항: {text}")
    response = respond_to_command(text)
    print(response)
    pyautogui.typewrite(response)           # 응답을 채팅창에 입력


# 수정 필요 : 명령에 대한 응답을 제공하는 함수
def respond_to_command(command):
    # 예시: 특정 명령어에 대한 응답 처리
    if "날씨" in command:
        response = "오늘의 날씨는 맑습니다.☀️"
    elif "시간" in command:
        response = "현재 시간은 3시입니다.🕒"
    else:
        response = "요청하신 내용을 이해하지 못했습니다. 다시 한 번 말씀해주세요 🥲"
    return response


def listen_for_audio():
    recorder = AudioToTextRecorder()
    while True:
        recorder.text(process_command)



if __name__ == "__main__":

    # streamlit page configuration
    st.set_page_config(layout="centered", initial_sidebar_state="expanded")
    # 페이지 제목
    st.title("안녕하세요. 당신의 안전을 책임질 SelenaAI 입니다 😊")

    # CSS 스타일 정의
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        align-items: start;
        margin-bottom: 20px;
        width: 100%;
    }

    .chat-image {
        width: 100px;
        height: 150px;
        border-radius: 10%;
        margin: 0 15px;
        object-fit : cover;
    }
    .chat-image.ai {
        flex-direction: row-reverse;
        align: right;
        align-items: flex-start;
        justify-content: flex-end;
    }
    .chat-message {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
    }

    .chat-container.ai {
        flex-direction: row-reverse;
        text-align: auto;
    }         
                
    .chat-message.ai {
        margin-right: 0;
        text-align: auto;
    }   

    </style>
    """, unsafe_allow_html=True)


    #음성 및 채팅모드 선택
    # mode = st.radio("어떤 모드를 원하십니까? : ", ["🔊", "⌨️"])
    # st.session_state.mode = mode
    question = st.text_input("언제든 편하게 질문하세요", value=st.session_state.question, key="user_input")


    # 질문에 대한 답변을 생성하는 버튼
    if question and (st.button('답변') or question != st.session_state.get('previous_question', '')):
        st.session_state['previous_question'] = question
        answer = ask_gpt(question)  # GPT-3 모델을 호출하여 답변을 받습니다.

        st.session_state.chat_history.append(f"Question: {question}")
        st.session_state.chat_history.append(f"Answer: {answer}")
        st.session_state.question = ""  # 입력 필드 클리어

        # 음성 모드일 경우 음성으로 답변 읽기
        if st.session_state.mode == "voice":
            speak(answer)

            
        # 대화 내역 표시
        for message in st.session_state.chat_history:
            formatted_message = message.replace("\n", "<br>")  # 🔹 f-string 바깥에서 변환 처리!

            if message.startswith("Question:"):
                st.markdown(
                    f"""
                    <div class="chat-container">
                        <img src="data:image/jpeg;base64,{get_image_base64(user_avatar)}" class="chat-image">
                        <div class="chat-message">{formatted_message}</div>
                        <div class="button-group">
                            <button class="copy-btn" onclick="copyToClipboard('{formatted_message}')"> ✔️ </button>
                            <button class="eval-btn" onclick="evaluateResponse('Good')"> 👍 </button>
                            <button class="eval-btn" onclick="evaluateResponse('Bad')"> 👎 </button>    
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )   .        .        .     

            else:
                st.markdown(
                    f"""
                    <div class="chat-container ai">
                        <img src="data:image/jpeg;base64,{get_image_base64(ai_avatar)}" class="chat-image">
                        <div class="chat-message ai">{formatted_message}</div>
                        <div class="button-group">
                            <button class="copy-btn" onclick="copyToClipboard('{formatted_message}')"> ✔️ </button>
                            <button class="eval-btn" onclick="evaluateResponse('Good')"> 👍 </button>
                            <button class="eval-btn" onclick="evaluateResponse('Bad')"> 👎 </button>    
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )




       # 자동 스크롤 아래로
        st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
        # 자동 스크롤 위로 (예: 100px 위로 이동)
        st.markdown("<script>window.scrollBy(0, -100);</script>", unsafe_allow_html=True)

            
    else:
        st.error("Please enter a question.")



    # 자바스크립트 함수 추가 (복사 및 평가 기능)
    st.markdown("""
    <div id="scroll-top" onclick="window.scrollTo(0,0)">⬆️</div>
    <div id="scroll-bottom" onclick="window.scrollTo(0,document.body.scrollHeight)">⬇️</div>
    <script>
        function copyToClipboard(text) {
            navigator.clipboard.writeText(text).then(() => {
                alert('Copied to clipboard');
            });
        }

        function evaluateResponse(evaluation) {
            alert('You rated this response as: ' + evaluation);
        }
    </script>
    """, unsafe_allow_html=True)



