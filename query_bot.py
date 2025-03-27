# query_bot.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()

CHROMA_DB_DIR = "db/chroma_e3"

def load_vector_db(persist_dir=CHROMA_DB_DIR):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="elementary-math"
    )
    return vectordb

def ask_question(vectordb, question):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        ),
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain({"query": question})
    return result["result"]

if __name__ == "__main__":
    print("📂 벡터 DB 로드 중...")
    vectordb = load_vector_db()

    # 👉 여기서 질문 바꿔가며 테스트!
    question = "초등학교 3학년 수준에 맞게 분수의 크기 비교 방법을 알려줘."
    answer = ask_question(vectordb, question)

    print("📌 질문:", question)
    print("💡 답변:", answer)
