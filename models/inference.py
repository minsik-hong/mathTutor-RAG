import torch
import numpy as np
from models.dkt_plus import DKTPlus
from models.test_info import load_student_data, get_student_data

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

def inference(model, q_seq, r_seq, top_n=10):
    device = next(model.parameters()).device

    q_seq = torch.tensor([q_seq], dtype=torch.long).to(device)
    r_seq = torch.tensor([r_seq], dtype=torch.float).to(device)

    with torch.no_grad():
        y = model(q_seq, r_seq)
        next_prob = y[:, -1, :]

    probs = next_prob.squeeze().cpu().numpy()
    problem_ids = np.arange(len(probs))

    problem_prob_pairs = list(zip(problem_ids, probs))
    sorted_problem_prob_pairs = sorted(problem_prob_pairs, key=lambda x: x[1])

    return sorted_problem_prob_pairs[:top_n]

def get_lowest_prob_problems(model_path, csv_path, student_index=1, num_problems=1865, top_n=10):
    # Load the model
    model = load_model(model_path, num_problems)

    # Load the student data
    students_data = load_student_data(csv_path, num_problems)

    # Get the student data based on student index
    q_seq, r_seq = get_student_data(students_data, student_index)

    # Run inference on the selected student's data
    top_10_lowest_prob = inference(model, q_seq, r_seq, top_n=top_n)

    # Extract Problem IDs
    problem_ids = [problem_id for problem_id, _ in top_10_lowest_prob]
    return problem_ids


# if __name__ == '__main__':
#     model_path = '/Users/hongminsik/Desktop/mathRag/models/model_best.pth'  # Path to your model
#     num_problems = 1865  # Total number of problems in the dataset
#     csv_path = '/Users/hongminsik/Desktop/mathRag/i-scream/i-scream_test.csv'  # Path to your student data CSV

#     # Load the model
#     model = load_model(model_path, num_problems)

#     # Load the student data
#     students_data = load_student_data(csv_path, num_problems)

#     # Get the student data based on student index (e.g., 1st student)
#     student_index = 1  # Change this to the desired student's index (0-based)
#     q_seq, r_seq = get_student_data(students_data, student_index)

#     # Run inference on the selected student's data
#     top_10_lowest_prob = inference(model, q_seq, r_seq, top_n=10)

#     # Output the results
#     print(f"Top 10 problems with the lowest predicted probabilities for student {student_index + 1}:")
#     for problem_id, prob in top_10_lowest_prob:
#         print(f"Problem ID: {problem_id}, Predicted Probability: {prob:.4f}")

