import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(file_path='intents.json'):
    with open(file_path) as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    patterns, labels = [], []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            labels.append(intent['tag'])
    return patterns, labels
