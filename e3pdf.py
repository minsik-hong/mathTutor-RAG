import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔐 Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()

# 📁 파일 경로
PDF_FILE = "data/e3.pdf"
CHROMA_DB_DIR = "db/chroma_e3"

# 1️⃣ PDF 문서 로드
def load_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

# 2️⃣ 문서 분할 - 수학 교과서에 맞게 문맥 단위로 자르기
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "?", "!", "⋯⋯", "▶", "활동", "예제", "문제", "식", "=", "+", "-"]
    )
    return text_splitter.split_documents(docs)

# 3️⃣ 문서를 벡터 DB에 저장
def save_to_vector_db(docs, persist_dir=CHROMA_DB_DIR):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(
        docs, embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="elementary-math"
    )
    vectordb.persist()
    return vectordb

# 4️⃣ 저장된 벡터 DB 로드
def load_vector_db(persist_dir=CHROMA_DB_DIR):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="elementary-math"
    )
    return vectordb

# 5️⃣ 질문에 답변하기
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
    # Step 1: 문서 로드
    print("📄 PDF 로드 중...")
    documents = load_pdf(PDF_FILE)
    print(f"총 {len(documents)}개의 raw 문서가 로드됨.")

    # Step 2: 문서 분할
    print("🧠 문서 분할 중...")
    split_docs = split_documents(documents)
    print(f"➡️ 총 {len(split_docs)}개의 문서 chunk 생성됨.")

    # Step 3: 벡터 DB 저장
    vectordb = save_to_vector_db(split_docs)
    print("✅ 벡터 DB 저장 완료")

    # Step 4: 질문 테스트
    question = "초등학교 3학년 수준에 맞도록, 곱셈에 대한 설명과 예제를 알려줘."
    answer = ask_question(vectordb, question)
    print("📌 질문:", question)
    print("💡 답변:", answer)
