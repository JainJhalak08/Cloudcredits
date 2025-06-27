from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Load dataset
df = pd.read_csv("sentiment_data.csv")
df.columns = df.columns.str.strip()  

# Prepare input and target
X = df["text"]
y = df["airline_sentiment"]

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Train model with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_vec, y)

# Check if model can predict 'neutral'
test_text = "The flight departed on time and landed safely."
test_vec = vectorizer.transform([test_text])
print("Test prediction (neutral check):", model.predict(test_vec)[0])

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form["text"]
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("sentiment_model.pkl")
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)
