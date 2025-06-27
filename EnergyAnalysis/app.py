from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)

# Load the CSV file and parse the correct column
df = pd.read_csv('energy_data.csv', parse_dates=['Datetime'])

# Create 'hour' feature from datetime
df['hour'] = df['Datetime'].dt.hour + df['Datetime'].dt.day * 24

# Prepare features and target
X = df[['hour']]
y = df['AEP_MW']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'energy_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hour = int(request.form['hour'])
    model = joblib.load('energy_model.pkl')
    prediction = model.predict([[hour]])
    return jsonify({'predicted_consumption': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
