# main.py
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # 최신 버전 사용
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
CHROMA_DB_DIR = "db/chroma_math_all"

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


# 🔄 질문 응답 함수 (Memory 포함)
def ask_question(vectordb, question: str):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    # 🔹 이전 대화 내역 불러오기
    history = memory.chat_memory.messages  # List[HumanMessage | AIMessage]

    # 🔹 대화 내역을 문자열로 변환
    history_text = ""
    for msg in history:
        role = "학생" if msg.type == "human" else "선생님"
        history_text += f"{role}: {msg.content}\n"

    # 🔸 프롬프트 템플릿
    prompt_template = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=(
            "당신은 수학 선생님입니다. 학생이 이해하기 쉽게 설명해주세요.\n\n"
            "다음은 수학 교과서에서 발췌한 내용입니다. 이 내용을 참고하여 질문에 답변하세요. \n\n"
            "답변은 반드시 아래 문서를 기반으로 하며, 문서에 없는 내용에 대해 답할 때는 초3~중3 수준의 수학 범위 내용을 알고 있다고 대답해.\n\n"
            "모든 답변에는 답변의 출처를 표기하세요.\n\n"
            "💬 이전 대화:\n{history}\n\n"
            "📘 교과서 내용:\n{context}\n\n"
            "🙋‍♀️ 질문:\n{question}\n\n"
            "🧠 답변:"
        )
    )

    # 🔸 LLM 구성
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY
    )

    # 🔸 체인 구성 (context + 대화 + 질문 전달)
    chain = (
        {
            "context": lambda x: "\n".join(doc.page_content for doc in x["docs"]),
            "question": lambda x: x["question"],
            "history": lambda x: x["history"]
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # 🔸 실행 + memory 업데이트
    result = chain.invoke({"docs": docs, "question": question, "history": history_text})
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(result)

    return result


# 🚀 서버 시작 시 벡터 DB 로드
vectordb = load_vector_db()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



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