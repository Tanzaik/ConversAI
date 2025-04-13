import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Sample training data
training_sentences = [
    "hello", "hi", "hey", "howdy",
    "what is your name", "who are you",
    "how are you", "how are things",
    "goodbye", "see you", "exit"
]
training_labels = [
    "greeting", "greeting", "greeting", "greeting",
    "identity", "identity",
    "status", "status",
    "goodbye", "goodbye", "goodbye"
]

# Encode labels
label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)

# Tokenize input
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=padded_sequences.shape[1]),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(set(training_labels_encoded)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, training_labels_encoded, epochs=500, verbose=0)

# Save for later
model.save("intent_model.h5")
import pickle
with open("tokenizer.pkl", "wb") as f: pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f: pickle.dump(label_encoder, f)
