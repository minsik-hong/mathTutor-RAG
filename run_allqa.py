import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from models.inference import get_lowest_prob_problems


# ✅ API 키 로딩
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
CHROMA_DB_DIR = "db/chroma_math_json"

# ✅ 벡터 DB 로드
def load_vector_db(persist_dir=CHROMA_DB_DIR):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# ✅ json 메타데이터 기반 질문
def ask_question_by_chapter_id(vectordb, chapter_id, question):
    retriever = vectordb.as_retriever(
        # k=1 -> 가장 유사한 문서 1개만 검색
        search_kwargs={"k": 1, "filter": {"chapter_id": str(chapter_id)}} # 이 곳에 참조 키 입력
    )

    prompt_template = """
당신은 초등학, 중학교 수학 교과 과정을 잘 아는 교육 전문가입니다.

다음 문서는 id가 {chapter_id}인 개념에 대한 설명, 학기 정보, 관련 단원, 성취 기준 등을 담고 있습니다.

학생이 이 개념을 어려워합니다. 아래 문서를 참고하여:
1. 개념을 간단하고 명확하게 설명하고,
2. 이 개념이 포함된 단원과 목차 흐름 속에서 어떤 위치인지 소개하며,
3. 성취 기준이 의미하는 내용을 풀어 설명하고,
4. 학생이 연습해볼 수 있는 실생활 기반의 문제를 한두 개 제시하고,
5. 문제에 대한 수식 작성 시 **LaTeX 수식**으로 작성해주세요. 수식은 `$...$` 또는 `$$...$$` 로 감싸 주세요.

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
""".replace("{chapter_id}", str(chapter_id))

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": question})
    answer = result["result"]

    # 🔍 관련 메타데이터에서 achievement_name 추출
    source_doc = result["source_documents"][0]
    achievement_name = source_doc.metadata.get("achievement_name", "정보 없음")

    return answer, achievement_name

# achievement_name 기반 문제 찾기
def ask_question_by_achievement_name(vectordb, achievement_name, question):
    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 1,  # 가장 유사한 하나의 문서만 가져옴
        }
    )

    # 검색 질의 자체에 achievement_name을 포함시켜 유사도 기반 검색 유도
    full_query = f"{achievement_name} 관련 수학 문제와 해설: {question}"

    prompt_template = """
당신은 초등학교 및 중학교 수학 문제를 잘 설명하는 교사입니다.

아래 문서는 특정 성취기준에 해당하는 수학 문제입니다.

학생이 이 문제를 이해하지 못하고 있습니다. 아래 문서를 참고하여:
1. 문제를 쉽게 풀어주는 해설을 제공하고,
2. 필요한 개념 설명을 함께 해주고,
3. 정답과 이유를 설명해 주세요.
4. 문제에 대한 수식 작성 시 **LaTeX 수식**으로 작성해주세요. 수식은 `$...$` 또는 `$$...$$` 로 감싸 주세요.

문제:
{context}

질문:
{question}
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": question})
    answer = result["result"]

    # source 문서에서 문제 이미지 및 주제 가져오기
    source_doc = result["source_documents"][0]
    image_path = source_doc.metadata.get("image_path")
    question_topic = source_doc.metadata.get("question_topic")

    # 이미지가 있다면 답변에 포함 (상대 경로로 변환)
    if image_path:
        # Markdown이 저장될 genResult 폴더 기준으로 상대 경로 계산
        rel_image_path = os.path.relpath(image_path, start="genResult")
        answer += f"\n\n**문제 이미지**:\n![문제 이미지]({rel_image_path})"


    # 문제 주제도 함께 제공
    if question_topic:
        answer = f"### 문제 주제: {question_topic}\n\n" + answer

    return answer

# ✅ Markdown 저장
def save_answer_as_markdown(question, answer, file_path="result.md", folder="genResult"):
    os.makedirs(folder, exist_ok=True)  # 폴더가 없으면 생성
    full_path = os.path.join(folder, file_path)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(f"# 질문\n\n{question}\n\n")
        f.write(f"# 답변\n\n{answer}\n")
    print(f"답변이 '{full_path}'에 저장되었습니다.")


# 메인 실행
if __name__ == "__main__":
    vectordb = load_vector_db()
    print("✅ 벡터 DB 로드 완료")

    # 🔹 Inference에서 Problem ID 가져오기
    model_path = "/Users/hongminsik/Desktop/mathRag/models/model_best.pth"
    csv_path = "/Users/hongminsik/Desktop/mathRag/i-scream/i-scream_test.csv"
    student_index = 1  # 분석할 학생의 인덱스
    num_problems = 1865  # 총 문제 수


    # Get Problem IDs with the lowest probabilities
    problem_ids = get_lowest_prob_problems(model_path, csv_path, student_index, num_problems, top_n=1) # 개수 조정
    print(f"🔹 가져온 Problem IDs: {problem_ids}")

    # 🔹 각 Problem ID에 대해 질문 생성 및 저장
    for problem_id in problem_ids:
        print(f"\nProcessing Problem ID: {problem_id}")

        # 1️⃣ 개념 설명 및 achievement_name 추출
        concept_question = f"학생이 개념 id {problem_id}에 해당하는 내용을 이해하지 못했습니다. 해당 개념에 대해 쉽게 설명해주고, 문제를 만들어주세요."
        concept_answer, achievement_name = ask_question_by_chapter_id(vectordb, problem_id, concept_question)
        print(f"추출된 성취 기준 이름: {achievement_name}")

        # 2️⃣ 성취 기준 기반 문제 예시 및 해설 생성
        if achievement_name != "정보 없음":
            problem_question = f"학생이 '{achievement_name}' 성취기준에 대한 문제를 이해하지 못하고 있습니다. 문제 해설과 개념 설명을 제공해주세요."
            problem_answer = ask_question_by_achievement_name(vectordb, achievement_name, problem_question)
        else:
            problem_answer = "관련된 성취 기준을 찾을 수 없어 문제 예시를 제공할 수 없습니다."

        # 3️⃣ 마크다운으로 저장
        output_file = f"result_concept_plus_problem_{problem_id}.md"
        os.makedirs("genResult", exist_ok=True)
        full_path = os.path.join("genResult", output_file)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(f"# 질문 (개념)\n\n{concept_question}\n\n")
            f.write(f"# 개념 관련 답변\n\n{concept_answer}\n\n")
            f.write(f"# 질문 (문제 예시)\n\n{problem_question}\n\n")
            f.write(f"# 문제 예시 및 해설\n\n{problem_answer}\n")

        print(f"✅ Problem ID {problem_id}에 대한 개념 + 문제 설명 저장 완료: {output_file}")

