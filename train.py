import json
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK data (run once)
nltk.download('punkt_tab')
nltk.download('punkt')

# Load dataset
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

tags = []
patterns = []

# Extract data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# NLP Preprocessing & Vectorization
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english', token_pattern=None)
X = vectorizer.fit_transform(patterns)
y = tags

# Model Training
model = LogisticRegression(random_state=0, max_iter=200)
model.fit(X, y)

# Save the trained model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model training complete! Files saved as model.pkl and vectorizer.pkl.")