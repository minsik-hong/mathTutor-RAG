# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from run_allqa import st_, load_vector_db, ask_question_by_chapter_id, ask_question_by_achievement_name

app = FastAPI()

# React와 연동을 위한 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시에는 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    q_input: List[int]
    r_input: List[int]

@app.post("/run-inference")
def run_inference(data: InferenceRequest):
    q_input = data.q_input
    r_input = data.r_input

    vectordb = load_vector_db()

    # 문제 추천
    problem_ids = st_(q_input, r_input, target_grade=3)

    results = []

    for problem_id in problem_ids:
        concept_question = f"학생이 개념 id {problem_id}에 해당하는 내용을 이해하지 못했습니다. 해당 개념에 대해 쉽게 설명해주고, 문제를 만들어주세요."
        concept_answer, achievement_name = ask_question_by_chapter_id(vectordb, problem_id, concept_question)

        if achievement_name != "정보 없음":
            problem_question = f"학생이 '{achievement_name}' 성취기준에 대한 문제를 이해하지 못하고 있습니다. 문제 해설과 개념 설명을 제공해주세요."
            problem_answer = ask_question_by_achievement_name(vectordb, achievement_name, problem_question)
        else:
            problem_answer = "관련된 성취 기준을 찾을 수 없어 문제 예시를 제공할 수 없습니다."

        results.append({
            "problem_id": problem_id,
            "concept_question": concept_question,
            "concept_answer": concept_answer,
            "achievement_name": achievement_name,
            "problem_question": problem_question,
            "problem_answer": problem_answer
        })

    return {"results": results}
