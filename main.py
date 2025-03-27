# main.py
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # 최신 버전 사용

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
CHROMA_DB_DIR = "db/chroma_e3"

# FastAPI 앱 초기화
app = FastAPI(title="수학 RAG 챗봇")

# CORS 설정 (필요시 Frontend 연동 시 사용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Vector DB 로드
def load_vector_db():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="elementary-math"
    )
    return vectordb

# 🔄 질문 응답 함수
def ask_question(vectordb, question: str):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)  # 최신 방식으로 교체

    # 🔸 프롬프트 템플릿
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "당신은 수학 선생님입니다. "
            "학생이 이해하기 쉽게 설명해주세요.\n\n"
            "배경 정보:\n{context}\n\n"
            "질문:\n{question}\n\n"
            "답변:"
        )
    )

    # 🔸 LLM 설정
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY
    )

    # 🔸 프롬프트 체인 구성
    chain = (
        {"context": lambda x: "\n".join(doc.page_content for doc in x["docs"]), "question": lambda x: x["question"]}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # 🔸 실행
    result = chain.invoke({"docs": docs, "question": question})
    return result

# 🚀 서버 시작 시 벡터 DB 로드
vectordb = load_vector_db()

# ✅ 요청 모델 정의
class QueryRequest(BaseModel):
    question: str

# 🌐 POST API 엔드포인트
@app.post("/ask")
async def ask(request: QueryRequest):
    question = request.question
    print(f"📥 질문 받음: {question}")
    answer = ask_question(vectordb, question)
    return {"question": question, "answer": answer}
