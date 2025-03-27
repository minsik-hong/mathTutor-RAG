# streamlit_chat.py
import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "http://localhost:8000/ask"  # FastAPI 백엔드 주소

st.set_page_config(page_title="수학 챗봇", layout="wide")
st.title("📚 수학 챗봇")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 질문 입력
user_input = st.text_input("질문을 입력하세요:", placeholder="예: 곱셈이 뭐예요?", key="input")

# 질문 버튼
if st.button("질문하기") and user_input.strip() != "":
    with st.spinner("답변 생성 중..."):
        response = requests.post(API_URL, json={"question": user_input})
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.session_state.chat_history.append(("🙋‍♀️ " + user_input, "🤖 " + answer))
        else:
            st.error("서버에서 응답을 받지 못했습니다.")

# 대화 기록 출력
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**{q}**")
    st.markdown(f"{a}")
    st.markdown("---")
