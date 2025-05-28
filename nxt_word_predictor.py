import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load corpus from file
with open('sample1.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts([corpus])
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Create input sequences 
input_sequences = []
for line in corpus.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i + 1])

# Padding 
max_len = max(len(seq) for seq in input_sequences)
padded_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Split predictors and label
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)

# Load or Train Model 
model_file = "nxt_word_model.h5"

if os.path.exists(model_file):
    model = load_model(model_file)
    print("Model loaded from disk.")
else:
    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len - 1))
    model.add(LSTM(150))
    model.add(Dense(vocab_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(X, y, epochs=100)
    model.save(model_file)
    print("Model trained and saved to disk.")


# Predict next words from user input 
text = input("Enter your starting text: ")

for _ in range(20):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs)

    # Get the predicted word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            text += ' ' + word
            print(text)
            time.sleep(0.5)
            break
