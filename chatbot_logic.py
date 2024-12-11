import numpy as np
from tensorflow.keras.models import load_model
import json
from model_training import predict_class
from preprocessing import load_intents, preprocess_data
import random

# Load your model and intents outside the function for performance
model = load_model('chatbot_model.h5')

# Load intents and preprocess data
intents = load_intents('intents.json')
words, classes, documents = preprocess_data(intents)

def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

def get_bot_response(user_message):
    # Preprocess the user message
    intents_list = predict_class(user_message, words, classes)  # Ensure predict_class function is defined properly
    intent_index = np.argmax(intents_list)  # Get the index of the highest probability
    intent = classes[intent_index]  # Get the corresponding intent

     # Debugging info
    print(f"User Message: {user_message}")
    print(f"Predicted Intent: {intent}, Confidence: {intents_list[0][intent_index]}")

    # Fetch the response from the intents
    response = get_response(intent)  # Ensure get_response function is defined properly
    return response


def get_response(intent):
    # Load your intents JSON file if needed
    with open('intents.json') as file:
        intents = json.load(file)

    # Loop through each intent to find the response
    for i in intents['intents']:
        if i['tag'] == intent:
            return np.random.choice(i['responses'])  # Return a random response from the list

    return "Sorry, I didn't understand that."
