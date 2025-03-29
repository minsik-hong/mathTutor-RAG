# 📦 설치 필요: pip install streamlit

import json
import streamlit as st
from models.inference3 import st_  # ✅ 함수 임포트

# 예시 문제 이미지 경로와 실제 정답 리스트
problem_image_paths = [
    "/Users/hongminsik/Desktop/mathRag/data/images/ele/3/P3_1_01_00040_29584.png",
    "/Users/hongminsik/Desktop/mathRag/data/images/ele/3/P3_1_01_00040_29584.png",
    "/Users/hongminsik/Desktop/mathRag/data/images/ele/3/P3_1_01_00040_29584.png"
]

# 실제 정답 리스트 (0 또는 1)
correct_answers = [1, 0, 1]

# 사용자 정답 리스트 저장용 (세션 상태 이용)
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

st.title("🧮 수학 문제 퀴즈")

# 모든 문제를 다 풀지 않았을 때
if st.session_state.current_index < len(problem_image_paths):
    index = st.session_state.current_index

    # 문제 이미지 보여주기
    st.image(problem_image_paths[index], caption=f"문제 {index + 1}")

    # 사용자 정답 입력
    answer = st.radio(
        "정답을 선택하세요:",
        options=[0, 1],
        key=f"answer_{index}"
    )

    if st.button("제출"):
        st.session_state.user_answers.append(answer)
        st.session_state.current_index += 1
        st.rerun()  # ✅ 여기로 수정!


else:
    st.success("✅ 모든 문제를 완료했습니다!")

    # 실제 정답과 비교
    result = [
        1 if user == correct else 0
        for user, correct in zip(st.session_state.user_answers, correct_answers)
    ]

    # 세션 상태에 저장
    st.session_state.result = result

    st.subheader("📊 결과 요약")
    st.write(f"🔹 사용자 정답: {st.session_state.user_answers}")
    st.write(f"🔹 실제 정답: {correct_answers}")
    st.write(f"✅ 채점 결과 (1: 정답, 0: 오답): {result}")

    # 🔹 st_ 함수로 후속 처리 실행
    q_input = [1874, 1873, 1876]  # 너가 사용한 문제 ID 리스트
    r_input = result

    problem_ids = st_(q_input, r_input, target_grade=3)

    st.subheader("📘 추천 문제 ID")
    st.write(f"🔍 선택된 문제 ID (낮은 확률): {problem_ids}")

    # 🔹 JSON으로도 저장 가능 (선택)
    with open("user_response.json", "w") as f:
        json.dump({
            "q_input": q_input,
            "r_input": r_input,
            "result": result,
            "selected_problems": problem_ids
        }, f)