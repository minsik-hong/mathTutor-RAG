# build_vector_db.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()

PDF_DIR = "data/pdfs"  # 여러 개 PDF가 들어 있는 폴더
CHROMA_DB_DIR = "db/chroma_math_all"

def load_all_pdfs(pdf_dir):
    documents = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            print(f"📄 {filename} - {len(docs)} 문서 로드됨")
            documents.extend(docs)
    return documents

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "?", "!", "활동", "예제", "식", "="]
    )
    return splitter.split_documents(docs)

def save_to_vector_db(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(
        docs, embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="elementary-math-all"
    )
    vectordb.persist()

if __name__ == "__main__":
    all_docs = load_all_pdfs(PDF_DIR)
    split_docs = split_documents(all_docs)
    print(f"🧩 총 {len(split_docs)}개의 chunk 생성")
    save_to_vector_db(split_docs)
    print("✅ 모든 PDF가 하나의 벡터 DB에 저장 완료")
