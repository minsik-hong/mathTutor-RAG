import pandas as pd
import json

# 1. CSV 파일에서 1번 열의 ID 불러오기
csv_path = "/Users/hongminsik/Desktop/mathRag/i-scream/knowledgeTag_skillID.csv"
df = pd.read_csv(csv_path)
existing_ids = set(df.iloc[:, 0])  # CSV 첫 번째 열 전체를 set으로 변환


# print(existing_ids)
print(len(existing_ids))

# 2. JSON 파일 불러오기
json_path = "/Users/hongminsik/Desktop/mathRag/data/jsons/[라벨]수학 지식체계 데이터 세트_210611.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. JSON에서 from/to ID 추출해서 set에 저장
json_ids = set()
for key, value in data.items():
    json_ids.add(value["fromConcept"]["id"])
    json_ids.add(value["toConcept"]["id"])

# print(json_ids)
print(len(json_ids))

# 4. CSV에는 있지만 JSON에는 없는 ID 찾기
missing_in_json = existing_ids - json_ids

# 5. 출력
print("❌ JSON에 없는 ID 목록 (CSV에는 있음):")
for mid in sorted(missing_in_json):
    print(mid)

print(f"\n📌 총 {len(missing_in_json)}개의 ID가 JSON에 존재하지 않습니다.")