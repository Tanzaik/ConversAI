import speech_recognition as sr
import pyttsx3
import tensorflow as tf
import numpy as np
import pickle

# Load model and NLP components
model = tf.keras.models.load_model("intent_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Intent-to-response map
responses = {
    "greeting": "Hi there! How can I help you?",
    "identity": "I'm ConversAI, your personal voice assistant.",
    "status": "I'm doing great, thanks for asking!",
    "goodbye": "Goodbye! Have a great day!"
}

# Init speech engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def speak(text):
    print(f"ConversAI: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            print(f"You: {user_input}")
            return user_input
        except sr.UnknownValueError:
            speak("Sorry, I didn’t catch that.")
            return ""
        except sr.RequestError:
            speak("There was a problem with the speech recognition service.")
            return ""

def get_intent(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=10, padding='post')
    predictions = model.predict(padded, verbose=0)
    label_index = np.argmax(predictions)
    return label_encoder.inverse_transform([label_index])[0]

def main():
    speak("Hello! I'm ConversAI. How can I assist you today?")
    while True:
        user_input = listen()
        if not user_input:
            continue
        if "exit" in user_input.lower() or "quit" in user_input.lower():
            speak("Exiting. Have a nice day!")
            break
        intent = get_intent(user_input)
        response = responses.get(intent, "I'm not sure how to respond to that.")
        speak(response)

if __name__ == "__main__":
    main()
