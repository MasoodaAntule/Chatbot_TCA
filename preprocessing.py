import json

def load_and_preprocess_content(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data["content"]  # Extract the 'content' key
