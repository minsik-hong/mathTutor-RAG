import torch
import pandas as pd
from models.dkt_plus import DKTPlus


def load_model(model_path, num_problems, emb_size=102, hidden_size=102, lambda_r=0.1, lambda_w1=0.03, lambda_w2=3.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DKTPlus(
        num_q=num_problems,
        emb_size=emb_size,
        hidden_size=hidden_size,
        lambda_r=lambda_r,
        lambda_w1=lambda_w1,
        lambda_w2=lambda_w2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_grade_indices(tag_path, target_grade):
    df = pd.read_csv(tag_path)
    grade_rows = df[df['grade'] == target_grade]
    indices = grade_rows.index.tolist()
    problem_ids = grade_rows['tag'].tolist()
    return indices, problem_ids

def inference(model, q_seq, r_seq, target_grade, top_n=10):
    device = next(model.parameters()).device

    q_seq = torch.tensor([q_seq], dtype=torch.long).to(device)
    r_seq = torch.tensor([r_seq], dtype=torch.float).to(device)

    with torch.no_grad():
        y = model(q_seq, r_seq)
        next_prob = y[:, -1, :]

    probs = next_prob.squeeze().cpu().numpy()

    tag_path = '/Users/hongminsik/Desktop/mathRag/i-scream/new_knowledge_tag.csv'

    # Load grade indices & problem ids
    grade_indices, grade_problem_ids = load_grade_indices(tag_path, target_grade)

    # Filter: remove excluded indices
    filtered = []
    exclude_indices = [982, 987, 1205, 344, 349, 352, 353, 368, 366, 369, 367, 376, 375, 382, 377, 387, 402, 395, 763, 764, 783,
               785, 803, 338, 791, 345, 818, 808, 816, 813, 814, 815, 1382, 1380, 1381, 841, 1549, 1363, 1317, 839,
               1318, 1321, 239, 1108, 97, 98, 1220, 1508, 1805, 1812, 1679, 141, 666, 678, 1340, 1704, 66, 1295, 1323,
               1326, 1296, 246, 264, 74, 1465, 1236, 1063, 1075, 1076, 88, 1496, 1498, 1503, 1025, 1031, 1221, 1229,
               727, 1235, 983, 1280, 1287, 354, 355, 583, 582, 1548, 1546, 1547, 102, 494, 495, 594, 365, 371, 372, 379,
               380, 374, 384, 390, 393, 392, 400, 394, 403, 404, 397, 1551, 1563, 464, 1747, 1746, 613, 1502, 617, 1857,
               1170, 1178, 534, 708, 1186, 343, 339, 341, 346, 351, 357, 370, 399, 1191, 1085, 768, 769, 804, 796, 806,
               817, 809, 1559, 1571, 1572, 1577, 1579, 1578, 1584, 1398, 773, 784, 1594, 1194, 1195, 1609, 156, 1292,
               1293, 492, 490, 491, 1798, 1808, 1828, 999, 1822, 1829, 1834, 1001, 1007, 1832, 931, 838, 857, 980, 1143,
               1490, 779, 794, 797, 819, 820, 812, 1399, 1400, 1329, 904, 1743, 958, 959, 960, 971, 1142, 1624, 1627,
               1477, 1484, 1487, 1636, 1610, 1814, 1659, 1672, 988, 995, 1049, 425, 1146, 431, 750, 756, 1497, 1499,
               1500, 1696, 1710, 1709, 1344, 998, 439, 569, 735, 753, 570, 571, 737, 740, 741, 742, 1078, 1501, 485,
               933, 1079, 1461, 1464]

    for idx, pid in zip(grade_indices, grade_problem_ids):
        if pid not in exclude_indices:
            filtered.append((pid, probs[idx]))

    if len(filtered) == 0:
        print(f"No valid problems left after filtering for grade {target_grade}.")
        return []

    # Sort by probability (ascending)
    filtered_sorted = sorted(filtered, key=lambda x: x[1])

    return filtered_sorted[:top_n]


def load_mapping(file_path = '/Users/hongminsik/Desktop/mathRag/i-scream/knowledgeTag_skillID.txt'):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split()
            mapping[int(value)] = int(key)
    return mapping

def convert_list(values, mapping):
    return [mapping.get(v, None) for v in values]

def convert_list_reverse(values, mapping):
    reverse_mapping = {v: k for k, v in mapping.items()}
    return [reverse_mapping.get(v, None) for v in values]

def st_(q_input, r_input, target_grade, model_path='/Users/hongminsik/Desktop/mathRag/models/model_best.pth',num_problems = 1865):
    file_path = '/Users/hongminsik/Desktop/mathRag/i-scream/knowledgeTag_skillID.txt'
    map_ = load_mapping(file_path)
    values = [int(x) for x in q_input]
    q_input = convert_list_reverse(values,map_)
    model = load_model(model_path, num_problems)
    top_10_lowest_prob=inference(model, q_input, r_input, target_grade,top_n=10)

    id_ = []

    for problem_id, _ in top_10_lowest_prob:
        id_.append(problem_id)



    return id_



if __name__ == '__main__':
    q = [1874,1873,1876,1877,461]
    r = [1,1,0,0,1]
    fa = st_(q,r,3)

    print(fa)



