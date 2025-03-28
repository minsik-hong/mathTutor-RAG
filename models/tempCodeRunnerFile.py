
if __name__ == '__main__':
    model_path = '/Users/hongminsik/Desktop/mathRag/models/model_best.pth'  # Path to your model
    num_problems = 1865  # Total number of problems in the dataset
    csv_path = '/Users/hongminsik/Desktop/mathRag/i-scream/i-scream_test.csv'  # Path to your student data CSV

    # Load the model
    model = load_model(model_path, num_problems)

    # Load the student data
    students_data = load_student_data(csv_path, num_problems)

    # Get the student data based on student index (e.g., 1st student)
    student_index = 1  # Change this to the desired student's index (0-based)
    q_seq, r_seq = get_student_data(students_data, student_index)

    # Run inference on the selected student's data
    top_10_lowest_prob = inference(model, q_seq, r_seq, top_n=10)

    # Output the results
    print(f"Top 10 problems with the lowest predicted probabilities for student {student_index + 1}:")
    for problem_id, prob in top_10_lowest_prob:
        print(f"Problem ID: {problem_id}, Predicted Probability: {prob:.4f}")

