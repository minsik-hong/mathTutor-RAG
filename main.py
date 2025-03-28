import os
from dotenv import load_dotenv

# ✅ 최신 import 경로 반영
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()

# JSON 파일 경로
JSON_FILE = "data/[라벨]수학 지식체계 데이터 세트_210611.json"
CHROMA_DB_DIR = "db/chroma_math_json"

# 1. JSON 문서 로드 (fromConcept / toConcept 각각 문서화)
# 1. JSON 문서 로드 (fromConcept / toConcept 각각 문서화)
def load_json(path):
    import json
    from langchain.schema import Document

    def safe_str(value):
        if value is None:
            return ""
        return str(value)

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = []

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
                "parent_id": safe_str(concept_id)
            }

            documents.append(Document(page_content=content, metadata=metadata))

    return documents



# 2. 문서를 벡터 DB에 저장
def save_to_vector_db(docs, persist_dir=CHROMA_DB_DIR):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

# 3. 저장된 벡터 DB 로드
def load_vector_db(persist_dir=CHROMA_DB_DIR):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vectordb

# 4. 질문에 답변하기
def ask_question_by_id(vectordb, concept_id, question):
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 1,
            "filter": {
                "concept_id": str(concept_id)
            }
        }
    )

    # ✅ 문자열 안에서 concept_id만 직접 삽입
    prompt_template = """
당신은 초등학교 수학 교과 과정을 잘 아는 교육 전문가입니다.

다음 문서는 id가 {concept_id}인 개념에 대한 설명, 학기 정보, 관련 단원, 성취 기준 등을 담고 있습니다.

학생이 이 개념을 어려워합니다. 아래 문서를 참고하여:
1. 개념을 간단하고 명확하게 설명하고,
2. 이 개념이 포함된 단원과 목차 흐름 속에서 어떤 위치인지 소개하며,
3. 성취 기준이 의미하는 내용을 풀어 설명하고,
4. 학생이 연습해볼 수 있는 실생활 기반의 문제를 한두 개 제시하고,
5. LaTeX 수식이 포함된 해설을 제공합니다.

문서:
{context}

질문:
{question}

다음 형식을 지켜주세요:

### 🧠 개념 설명
...

### 📘 단원과 목차 정보
...

### 🎯 성취 기준 해설
...

### 🧩 연습 문제
문제 1: ...
문제 2: ...

### ✅ 정답 및 해설
정답 1: ...
정답 2: ...
"""


    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": question})
    return result["result"]



# 5. 결과를 마크다운으로 저장
def save_answer_as_markdown(question, answer, file_path="result.md"):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# 📌 질문\n\n{question}\n\n")
        f.write(f"# 💡 답변\n\n{answer}\n")
    print(f"📄 답변이 '{file_path}'에 저장되었습니다.")

# 메인 실행
if __name__ == "__main__":
    # Step 1: JSON 로드
    documents = load_json(JSON_FILE)
    print(f"총 {len(documents)}개의 문서 조각이 로드되었습니다.")

    # Step 2: 벡터 DB에 저장
    vectordb = save_to_vector_db(documents)
    print("✅ 벡터 DB 저장 완료")

    # Step 3: 질문 테스트
    concept_id = 5844
    question = "학생이 개념 id 5844에 해당하는 내용을 이해하지 못했습니다. 쉽게 설명해주고 문제를 만들어주세요."
    answer = ask_question_by_id(vectordb, concept_id, question)



    # Step 4: 출력 및 저장
    print("📌 질문:", question)
    print("💡 답변:", answer)
    save_answer_as_markdown(question, answer)
