# ConversAI
TLDR; voice-activated AI chatbot

ConversAI is a real-time, voice-activated AI chatbot powered by TensorFlow, NLP, and text-to-speech technology. Built with Python, the system listens to user speech, classifies intent using a trained model, and responds with intelligent, spoken replies — enabling natural and intuitive human-computer interaction.

🚀 Features
🎙️ Voice Input via microphone using SpeechRecognition

🤖 Intent Classification with a custom TensorFlow NLP model

🔊 Text-to-Speech Output using pyttsx3

💬 Handles greetings, identity, status checks, and goodbyes

🔁 Real-time, interactive conversation loop

🛠 Tech Stack
Python 3.8+

TensorFlow – NLP model for intent recognition

SpeechRecognition – Transcribes real-time microphone input

pyttsx3 – Converts AI responses to speech

scikit-learn – Encodes label categories

🧪 Sample Intents
User Says	ConversAI Responds
"Hello"	"Hi there! How can I help you?"
"What's your name?"	"I'm ConversAI, your personal voice assistant."
"How are you?"	"I'm doing great, thanks for asking!"
"Bye"	"Goodbye! Have a great day!"
🧰 How to Run
Install dependencies:

bash
Copy
Edit
pip install tensorflow speechrecognition pyttsx3 scikit-learn numpy
Train the NLP model: Run train_intent_model.py to generate:

intent_model.h5

tokenizer.pkl

label_encoder.pkl

Start the chatbot:

bash
Copy
Edit
python conversai.py
📂 Project Structure
bash
Copy
Edit
ConversAI/
├── conversai.py            # Main voice loop & response system
├── train_intent_model.py   # Script to train intent classification model
├── intent_model.h5         # Saved NLP model
├── tokenizer.pkl           # Tokenizer used for preprocessing
├── label_encoder.pkl       # Encoded label classes
└── README.md               # Project documentation


Future Enhancements??
Integrate BERT or Transformer-based NLP for deeper understanding

Add support for more intents via JSON configuration

Build a GUI interface with Tkinter or PyQt

Deploy as a web app using Flask or Streamlit


ConversAI showcases the power of combining speech recognition, NLP, and real-time interaction to create intelligent, voice-driven applications. Whether as a foundation for more complex assistants or a proof-of-concept for natural language systems, this project demonstrates how accessible and scalable conversational AI can be with the right tools. Contributions and improvements are always welcome!
