import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
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

# ✅ chapter_id 기반 질문
def ask_question_by_chapter_id(vectordb, chapter_id, question):
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 1, "filter": {"chapter_id": str(chapter_id)}}
    )

    prompt_template = """
당신은 초등학교 수학 교과 과정을 잘 아는 교육 전문가입니다.

다음 문서는 id가 {chapter_id}인 개념에 대한 설명, 학기 정보, 관련 단원, 성취 기준 등을 담고 있습니다.

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
""".replace("{chapter_id}", str(chapter_id))

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.3, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": question})
    return result["result"]

# ✅ 문제 ID 기반 질문 (문제 ID는 문자열)
def ask_question_by_problem_id(vectordb, problem_id, question):
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 1, "filter": {"id": str(problem_id)}}
    )

    prompt_template = """
당신은 초등학교 및 중학교 수학 문제를 잘 설명하는 교사입니다.

다음 문서는 id가 {problem_id}인 실제 수학 문제에 대한 텍스트 설명입니다.

학생이 문제를 이해하지 못하고 있습니다. 아래 문서를 참고하여:
1. 문제를 쉽게 풀어주는 해설을 제공하고,
2. 필요한 개념 설명을 함께 해주고,
3. 정답과 이유를 설명해 주세요.

문제:
{context}

질문:
{question}
""".replace("{problem_id}", str(problem_id))

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

    # ✅ 이미지가 있을 경우 포함
    source_doc = result["source_documents"][0]
    image_path = source_doc.metadata.get("image_path")
    if image_path:
        answer += f"\n\n![문제 이미지]({image_path})"

    return answer

# ✅ Markdown 저장
def save_answer_as_markdown(question, answer, file_path="result.md"):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# 📌 질문\n\n{question}\n\n")
        f.write(f"# 💡 답변\n\n{answer}\n")
    print(f"📄 답변이 '{file_path}'에 저장되었습니다.")

# ✅ 메인 실행
if __name__ == "__main__":
    vectordb = load_vector_db()
    print("✅ 벡터 DB 로드 완료")

    # 🔹 Inference에서 Problem ID 가져오기
    model_path = "/Users/hongminsik/Desktop/mathRag/models/model_best.pth"
    csv_path = "/Users/hongminsik/Desktop/mathRag/i-scream/i-scream_test.csv"
    student_index = 1  # 분석할 학생의 인덱스
    num_problems = 1865  # 총 문제 수


    # Get Problem IDs with the lowest probabilities
    problem_ids = get_lowest_prob_problems(model_path, csv_path, student_index, num_problems, top_n=10)
    print(f"🔹 가져온 Problem IDs: {problem_ids}")

    # 🔹 각 Problem ID에 대해 질문 생성 및 저장
    for problem_id in problem_ids:
        question = f"학생이 개념 id {problem_id}에 해당하는 내용을 이해하지 못했습니다. 해당 개념에 대해 쉽게 설명해주고, 문제를 만들어주세요."
        answer = ask_question_by_chapter_id(vectordb, problem_id, question)
        output_file = f"result_concept_{problem_id}.md"
        save_answer_as_markdown(question, answer, output_file)
        print(f"✅ Problem ID {problem_id}에 대한 답변 저장 완료: {output_file}")

    # # 🔹 chapter_id = 528
    # chapter_id = 414
    # question1 = "학생이 개념 id 2773에 해당하는 내용을 이해하지 못했습니다. 해당 개념에 대해 쉽게 설명해주고, 문제를 만들어주세요."
    # answer1 = ask_question_by_chapter_id(vectordb, chapter_id, question1)
    # save_answer_as_markdown(question1, answer1, "result_concept_2773.md")

    # # 🔹 문제 ID = "25763_84170"
    # problem_id = "25763_84170"
    # question2 = "문제 ID 25763_84170번을 학생이 이해하지 못했습니다. 문제 해설과 개념 설명을 해주세요."
    # answer2 = ask_question_by_problem_id(vectordb, problem_id, question2)
    # save_answer_as_markdown(question2, answer2, "result_problem_25763_84170.md")
