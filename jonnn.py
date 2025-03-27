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
JSON_FILE = "data/P5_1_01_00042_30429.json"
CHROMA_DB_DIR = "db/chroma_math_json"

# 1. JSON 문서 로드
def load_json(path):
    jq_schema = '''
    . as $item
    | {
        page_content: ($item.OCR_info[0].question_text),
        metadata: {
            id: $item.id,
            grade: $item.question_info[0].question_grade,
            unit: $item.question_info[0].question_unit,
            topic_name: $item.question_info[0].question_topic_name,
            difficulty: $item.question_info[0].question_difficulty
        }
    }
    '''
    loader = JSONLoader(
        file_path=path,
        jq_schema=jq_schema,
        text_content=False
    )
    return loader.load()

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
def ask_question(vectordb, question):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # ✅ PromptTemplate에 context 포함
    prompt_template = """
당신은 수학 교육을 위한 전문가입니다. 아래 문서를 참고하여 학생에게 적절한 수학 문제를 생성하고, 명확한 정답을 LaTeX 형식으로 작성하세요.

문서:
{context}

질문:
{question}

아래와 같은 형식을 따르세요:

문제:
...

답:
(LaTeX 수식 포함)
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

    result = qa_chain({"query": question})
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
    question = "곱셈과 나눗셈이 섞여 있는 식의 계산문제 내줘"
    answer = ask_question(vectordb, question)

    # Step 4: 출력 및 저장
    print("📌 질문:", question)
    print("💡 답변:", answer)
    save_answer_as_markdown(question, answer)
