import os
import json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# ✅ 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()

# ✅ 폴더 경로
JSON_FOLDER = "data/jsons"
PDF_FOLDER = "data/pdfs"
IMG_JSON_FOLDER = "data/image_jsons"
IMG_FOLDER = "data/images"
CHROMA_DB_DIR = "db/chroma_math_json"

# ✅ 안전한 문자열 변환
def safe_str(value):
    return "" if value is None else str(value)

# ✅ 개념 설명 JSON 로딩
def load_json_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)

        # 
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

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
                    "source": filename
                }

                documents.append(Document(page_content=content, metadata=metadata))

    return documents

# ✅ PDF 문서 로딩
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

# ✅ 이미지 기반 JSON 로딩 (하위 폴더 포함)
def load_image_json_documents(folder_path, image_folder):
    documents = []

    for root, _, files in os.walk(folder_path):  # 하위 폴더까지 탐색
        for filename in files:
            if not filename.endswith(".json"):
                continue

            file_path = os.path.join(root, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # OCR 텍스트 가져오기
            question_text = data["OCR_info"][0].get("question_text", "")
            image_filename = data.get("question_filename", "")

            # 이미지 경로 찾기 (image_folder 하위 전체에서 검색)
            image_path = None
            for img_root, _, img_files in os.walk(image_folder):
                if image_filename in img_files:
                    image_path = os.path.join(img_root, image_filename)
                    break

            info = data.get("question_info", [{}])[0]  # 첫 번째 question_info 사용

            metadata = {
                "id": data.get("id", ""),
                "question_topic": info.get("question_topic_name", ""),
                "question_grade": info.get("question_grade", ""),
                "question_type": info.get("question_type1", ""),
                "question_topic_name": info.get("question_topic_name", ""),
                "difficulty": info.get("question_difficulty", ""),
                "step": info.get("question_step", ""),
                "image_path": image_path or "경로 없음",
                "source": filename
            }

            documents.append(Document(page_content=question_text, metadata=metadata))

    return documents

# ✅ 벡터 DB에 저장
def save_to_vector_db(docs, persist_dir=CHROMA_DB_DIR):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

# ✅ 메인 실행
if __name__ == "__main__":
    json_docs = load_json_documents(JSON_FOLDER)
    print(f"✅ 개념 JSON 문서 {len(json_docs)}개 로드 완료")

    pdf_docs = load_pdf_documents(PDF_FOLDER)
    print(f"✅ PDF 문서 {len(pdf_docs)}개 로드 완료")

    image_json_docs = load_image_json_documents(IMG_JSON_FOLDER, IMG_FOLDER)
    print(f"✅ 이미지 기반 문제 문서 {len(image_json_docs)}개 로드 완료")

    all_docs = json_docs + pdf_docs + image_json_docs
    print(f"📚 총 문서 수: {len(all_docs)}")

    save_to_vector_db(all_docs)
    print("✅ 벡터 DB 저장 완료")
