from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from utils import load_data, preprocess_data
import numpy as np

def train_chatbot():
    # Load and preprocess data
    data = load_data()
    patterns, labels = preprocess_data(data)
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(patterns)
    X = tokenizer.texts_to_sequences(patterns)
    X = pad_sequences(X, maxlen=10)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    y = np.array(y)

    # Build the model
    model = Sequential([
        Dense(128, input_shape=(X.shape[1],), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(set(labels)), activation='softmax')
    ])

    # Compile and train the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5')
    print("Model trained and saved as chatbot_model.h5")

if __name__ == "__main__":
    train_chatbot()
