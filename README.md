# Chatbot-Project
- 목적 : 해당 AI는 운전자의 실시간 음성 처리와 그에 따른 자연스러운 응답을 제공하는 챗봇입니다. 이름은 'SelenaAI'입니다. 사용자 중심의 편리한 대화 경험을 제공하기 위해, 실시간 음성인식(STT : Speech To Text) 및 변환(TTS : Text To Seech)이 가능한 챗봇입니다.

- 활용 장비 및 재료
  1) 서버 : Vultr, Gabia를 활용한 Ubuntu 환경
  2) 언어 : Python3 및 Streamlit
  3) 라이브러리 : faster_whisper(STT), gTTS(TTS), OpenAI API(API) 등
  4) 개발도구 : VScode

<br>
![SmartSelect_20250519_120927_Slides](https://github.com/user-attachments/assets/c93f7afd-25f0-4a5a-9793-7bc141095abb)
![SmartSelect_20250519_120944_Slides](https://github.com/user-attachments/assets/232b23e1-1725-44ba-894a-a050eabd54f6)
![SmartSelect_20250519_121005_Slides](https://github.com/user-attachments/assets/80b42ad9-bab9-4acd-9ce9-d00127d36c2e)

<br><br>
 
- 구조
  1) 라이브러리 임포트
  2) streamlit을 활용하여 환경설정
  3) html, css, js를 활용하여 UI 구성
  4) STT : faster_whisper 캐싱
  5) 챗봇 엔진 호출
  6) TTS : gTTS로 MP3 생성 및 재생(속도조절 가능)


 
- 활용방안 및 기대효과
  1) 일상생활을 지원하는 개인비서로 사용 가능
  2) 즉각적인 피드백으로, 음성기반 FAQ 응대 및 24/7 인공지능 상담원도 가능할 것

- 일정
  ![SmartSelect_20250519_115731_Slides](https://github.com/user-attachments/assets/b596e886-a73a-468f-9926-e32cd44d798e)


<br>

- 상세 정보
  1) 실시간 STT : record_audio함수를 사용하여 녹은된 사용자 음성을 faster_whisper(medium)으로 텍스트 변환
  2) TTS : gTTS로 음성 파일 생성한 후, Streamlit으로 채팅 메세지 및 오디오 음성 출력
  3) OpenAI GPT-4-turbi : 자연어 처리 및 응답 생성




- 피드백
  1) Pros : gTTS와 Streamlit을 사용하여, cpu 사용량을 줄였다. 그리고 음성인식할 때, 버튼으로 녹음을 시작하여 음성 인식률을 높였습니다.
  2) Cons : gTTS는 음성이 다소 부자연스러워서, 'edge_tts'를 사용하는 것을 추천합니다. 아니면 '일레븐렙스'를 사용하여, 유료로 자연스러운 응답을 생성하는 것도 좋은 대안일 것 같습니다 😄
