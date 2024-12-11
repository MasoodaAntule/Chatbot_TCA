import numpy as np
import random
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from preprocessing import preprocess_data, load_intents, lemmatizer


# Load the trained model (ensure the model is already trained and saved)
model = load_model("chatbot_model.h5")

def predict_class(message, words, classes):
    # Preprocess the message to create a bag of words
    bow_vector = [0] * len(words)
    message_words = [lemmatizer.lemmatize(word.lower()) for word in message.split()]
    
    for word in message_words:
        if word in words:
            bow_vector[words.index(word)] = 1
            
    # Make prediction
    res = model.predict(np.array([bow_vector]))[0]
    return [[i, r] for i, r in enumerate(res) if r > 0.5]  # Filter results

def prepare_training_data(words, classes, documents):
    training = []
    output_empty = [0] * len(classes)
    
    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        
        for w in words:
            bag.append(1) if w in word_patterns else bag.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)
    return np.array(list(training[:, 0])), np.array(list(training[:, 1]))

def build_and_train_model(train_x, train_y, classes):
    model = Sequential()
    model.add(Dense(128, input_shape=(train_x.shape[1],), activation='relu'))  # Correct input shape
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))
    
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    model.save("chatbot_model.h5")
    return model
