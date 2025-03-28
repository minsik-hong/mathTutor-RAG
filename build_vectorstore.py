# build_vectorstore.py
import os
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# ✅ 환경 변수에서 API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()

# ✅ 데이터 폴더 경로
JSON_FOLDER = "data/jsons"
PDF_FOLDER = "data/pdfs"
CHROMA_DB_DIR = "db/chroma_math_json"

# ✅ JSON 파일 로딩
def load_json_documents(folder_path):
    def safe_str(value):
        return "" if value is None else str(value)

    documents = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for concept_id, concept_data in raw_data.items():
            for key in ["fromConcept", "toConcept"]:
                concept = concept_data.get(key, {})
                if not concept:
                    continue

                content = f"{concept.get('name')}\n\n{concept.get('description', '')}"
                metadata = {
                    "concept_id": safe_str(concept.get("id")),
                    "semester": safe_str(concept.get("semester")),
                    "chapter_id": safe_str(concept.get("chapter", {}).get("id")),
                    "chapter_name": safe_str(concept.get("chapter", {}).get("name")),
                    "achievement_id": safe_str(concept.get("achievement", {}).get("id")),
                    "achievement_name": safe_str(concept.get("achievement", {}).get("name")),
                    "relation_type": key,
                    "parent_id": safe_str(concept_id),
                    "source": filename  # 파일 이름 메타데이터에 추가
                }

                documents.append(Document(page_content=content, metadata=metadata))

    return documents

# ✅ PDF 파일 로딩
def load_pdf_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, filename)
        loader = PyPDFLoader(file_path)
        pdf_docs = loader.load()

        for doc in pdf_docs:
            doc.metadata["source"] = filename

        documents.extend(pdf_docs)

    return documents

# ✅ 벡터 DB 저장
def save_to_vector_db(docs, persist_dir=CHROMA_DB_DIR):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

# ✅ 메인 실행
if __name__ == "__main__":
    json_docs = load_json_documents(JSON_FOLDER)
    print(f"✅ JSON 문서 {len(json_docs)}개 로드 완료")

    pdf_docs = load_pdf_documents(PDF_FOLDER)
    print(f"✅ PDF 문서 {len(pdf_docs)}개 로드 완료")

    all_docs = json_docs + pdf_docs
    print(f"📚 총 문서 수: {len(all_docs)}")

    save_to_vector_db(all_docs)
    print("✅ 벡터 DB 저장 완료")
