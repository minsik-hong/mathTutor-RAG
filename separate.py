import pandas as pd
import json

# 1. CSV 불러오기
csv_path = "/Users/hongminsik/Desktop/mathRag/i-scream/knowledgeTag_skillID.csv"
df = pd.read_csv(csv_path)

# 컬럼이 2개라고 가정: 컬럼 0 = ID, 컬럼 1 = 점수 또는 레벨 값
df.columns = ['skill_id', 'value']  # 필요시 컬럼명 조정
csv_id_to_value = dict(zip(df['skill_id'], df['value']))  # 딕셔너리로 빠르게 조회

# 2. JSON 불러오기
json_path = "/Users/hongminsik/Desktop/mathRag/data/jsons/[라벨]수학 지식체계 데이터 세트_210611.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. "semester": "초등-초1" 에 해당하는 ID들만 수집
target_ids = set()

for item in data.values():
    if item["fromConcept"]["semester"] in ["초등-초1-1학기", "초등-초1-2학기"]:
        target_ids.add(item["fromConcept"]["id"])
        target_ids.add(item["toConcept"]["id"])

# 4. CSV에서 해당 ID들에 대응하는 값들 추출
matched_values = []
for id_ in target_ids:
    if id_ in csv_id_to_value:
        matched_values.append(csv_id_to_value[id_])

# 5. 최댓값 찾기
if matched_values:
    max_value = max(matched_values)
    print(f"✅ '초등-초1-1학기,2학기'에 해당하는 ID 중, CSV에서 매칭된 값들의 최댓값은: {max_value}")
else:
    print("⚠️ 해당 조건에 맞는 ID가 CSV에서 하나도 매칭되지 않았습니다.")
