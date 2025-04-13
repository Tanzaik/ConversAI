import speech_recognition as sr
import pyttsx3
import tensorflow as tf
import numpy as np
import pickle

# Load model & encoders
model = tf.keras.models.load_model("intent_model.h5")
with open("tokenizer.pkl", "rb") as f: tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f: label_encoder = pickle.load(f)

# Responses
responses = {
    "greeting": "Hi there! How can I help you?",
    "identity": "I'm ConversAI, your personal voice assistant.",
    "status": "I'm doing great, thanks for asking!",
    "goodbye": "Goodbye! Have a great day!"
}

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except:
            print("Sorry, I didn’t catch that.")
            return ""

def get_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=10, padding='post')
    pred = model.predict(padded, verbose=0)
    label_index = np.argmax(pred)
    return label_encoder.inverse_transform([label_index])[0]

def main():
    speak("Hi, I'm ConversAI. What would you like to talk about?")
    while True:
        user_input = listen()
        if not user_input:
            continue
        if "exit" in user_input.lower():
            speak("Goodbye!")
            break
        intent = get_intent(user_input)
        response = responses.get(intent, "Sorry, I didn't understand that.")
        speak(response)

if __name__ == "__main__":
    main()
