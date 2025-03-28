import numpy as np


def load_student_data(csv_path, num_problems):
    students_data = []

    # Open the CSV file and read it line by line
    with open(csv_path, 'r') as file:
        lines = file.readlines()

        # Every 3 lines corresponds to one student: ID, problem codes, responses
        for i in range(0, len(lines), 3):
            student_id = lines[i].strip().split(',')  # First row (student IDs), no need to convert to int
            problem_codes = list(map(int, lines[i + 1].strip().split(',')))  # Second row (problem codes)
            responses = list(map(int, lines[i + 2].strip().split(',')))  # Third row (responses)

            # Process each student's data into sequences of questions and responses
            students_data.append((problem_codes, responses))

    return students_data


def get_student_data(students_data, student_index):
    # Return question_id and response sequence for a specific student
    q_seq, r_seq = students_data[student_index]
    return q_seq, r_seq
