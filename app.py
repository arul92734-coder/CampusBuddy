from flask import Flask, render_template, request, jsonify
import json
import pickle
import random
import nltk  # ← add this

# Download NLTK data  ← add these
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

app = Flask(__name__)
# Load model, vectorizer, and intents
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    # Process user input
    user_message_vec = vectorizer.transform([user_message])
    prediction = model.predict(user_message_vec)[0]
    
    # Generate response
    response = "I'm sorry, I didn't understand that. Could you rephrase?"
    for intent in intents['intents']:
        if intent['tag'] == prediction:
            response = random.choice(intent['responses'])
            break
            
    return jsonify({"response": response})
@app.route('/health')
def health():
    return "ok"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)