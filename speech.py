import speech_recognition as sr
import pyttsx3
import tensorflow as tf  # Placeholder for NLP model
import numpy as np        # If needed by your model

# Initialize recognizer and TTS
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Simulated NLP logic (replace with TensorFlow model later)
def nlp_response(user_input):
    responses = {
        "hello": "Hi there! How can I help you?",
        "what's your name": "I'm ConversAI, your personal voice assistant.",
        "how are you": "I'm doing great, thank you for asking!",
    }
    for key in responses:
        if key in user_input.lower():
            return responses[key]
    return "Sorry, I didn't quite catch that. Can you repeat?"

# Speak the response
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Listen to the microphone
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            return ""
        except sr.RequestError:
            print("Speech service error.")
            return ""

# Main loop
def main():
    speak("Hello, I am ConversAI. How can I assist you today?")
    while True:
        user_input = listen()
        if user_input:
            if "exit" in user_input.lower() or "quit" in user_input.lower():
                speak("Goodbye!")
                break
            response = nlp_response(user_input)
            speak(response)

if __name__ == "__main__":
    main()
