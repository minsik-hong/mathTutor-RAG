# 📦 설치 필요: pip install streamlit

import json
import streamlit as st
from models.inference3 import st_  # ✅ 함수 임포트
from run_allqa import st_2, load_vector_db

# 예시 문제 이미지 경로
problem_image_paths = [
    "/Users/hongminsik/Desktop/mathRag/eximage/428.png",
    "/Users/hongminsik/Desktop/mathRag/eximage/454.png",
    "/Users/hongminsik/Desktop/mathRag/eximage/1456.png",
    "/Users/hongminsik/Desktop/mathRag/eximage/2615.png",
    "/Users/hongminsik/Desktop/mathRag/eximage/2616.png",
    "/Users/hongminsik/Desktop/mathRag/eximage/2617.png",
    "/Users/hongminsik/Desktop/mathRag/eximage/2618.png",
    "/Users/hongminsik/Desktop/mathRag/eximage/2619.png",
    "/Users/hongminsik/Desktop/mathRag/eximage/5283.png"
]

# 보기 옵션 (문제 순서대로: 총 8문제)
options = [
    ["6$cm^2$", "12$cm^2$", "18$cm^2$", "24$cm^2$", "30$cm^2$"],  # 428.png
    ["85회", "86회", "87회", "88회", "89회"],                   # 454.png
    ["48세", "50세", "52세", "54세", "56세"],                   # 1456.png
    ["1,2,3", "1,2,4", "1,2,5", "1,3,5", "2,3,4"],                # 2615.png
    ["ㄱ,ㄴ,ㄷ,ㄹ", "ㄴ,ㄷ,ㄹ,ㄱ", "ㄴ,ㄷ,ㄱ,ㄹ", "ㄷ,ㄴ,ㄹ,ㄱ", "ㄷ,ㄴ,ㄱ,ㄹ"],  # 2616.png
    ["2대", "3대", "4대", "5대", "6대"],                         # 2617.png
    ["65/3", "70/3", "65/9", "70/9", "75/9"],                    # 2618.png
    ["0.3시간", "0.4시간", "0.5시간", "0.6시간", "0.7시간"],       # 2619.png
    ["1000명", "1100명", "1200명", "1300명", "1400명"]                # 5283.png
]

# 정답 리스트 (각 문제 정답 번호: 1부터 시작)
correct_answers = [2, 2, 1, 2, 5, 4, 2, 3, 3]


# 세션 상태 초기화
if "user_answers" not in st.session_state:
    st.session_state.user_answers = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

st.title("🧮 수학 역량 진단 문제 (5지 선다형)")

# 아직 안 푼 문제 있을 경우
if st.session_state.current_index < len(correct_answers):
    index = st.session_state.current_index

    # 문제 이미지 출력
    st.image(problem_image_paths[index], caption=f"문제 {index + 1}")

    # 문제별 보기 가져오기
    choices = options[index]

    # 보기 선택
    answer = st.radio(
        "정답을 선택하세요:",
        options=choices,
        key=f"answer_{index}"
    )

    # 제출 버튼
    if st.button("제출"):
        selected_idx = choices.index(answer) + 1  # 1~5로 정답 번호 환산
        st.session_state.user_answers.append(selected_idx)
        st.session_state.current_index += 1
        st.rerun()


else:
    st.success("✅ 모든 문제를 완료했습니다!")

    # 정답 채점
    result = [
        1 if user == correct else 0
        for user, correct in zip(st.session_state.user_answers, correct_answers)
    ]
    st.session_state.result = result

    st.subheader("📊 결과 요약")
    st.write(f"🔹 사용자 정답: {st.session_state.user_answers}")
    st.write(f"🔹 실제 정답: {correct_answers}")
    st.write(f"✅ 채점 결과 (1: 정답, 0: 오답): {result}")

    # st_ 함수 호출
    q_input = [428, 454, 1456, 2615, 2616, 2617, 2618, 2619, 5283]  # 문제 ID 예시
    r_input = result
    problem_ids = st_(q_input, r_input, target_grade=3)
    # problem_ids, achievement_standards = st_(q_input, r_input, target_grade=3)

    st.subheader("📘 추천 문제 ID")
    st.write(f"🔍 당신의 약점 코드: {problem_ids}")

    # st.subheader("🎯 약점 성취 기준")
    # for idx, standard in enumerate(achievement_standards, 1):
    #     st.markdown(f"**{idx}.** {standard}")

    # 결과 JSON 저장
    with open("user_response.json", "w") as f:
        json.dump({
            "q_input": q_input,
            "r_input": r_input,
            "result": result,
            "selected_problems": problem_ids
        }, f)

    CHROMA_DB_DIR = "db/chroma_math_json"
    vectordb = load_vector_db(persist_dir=CHROMA_DB_DIR)
    # st_2(vectordb, q_input, r_input)
