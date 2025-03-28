def load_txt_to_dict(file_path= '/home/hail/ze/data/i-scream/knowledgeTag_skillID.txt'):
    data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()  # Assuming space or tabs are the delimiter between ID and value
            if len(parts) == 2:  # Ensure there are two elements in the line
                id_value = parts[0]  # Left column (ID)
                value = parts[1]     # Right column (value)
                data_dict[id_value] = value  # Store in dictionary

    return data_dict


def get_value_by_id(data_dict, id_value):
    return data_dict.get(id_value, "ID not found")  # Return value if ID exists, else a default message



