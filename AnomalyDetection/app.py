from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)

def preprocess(df):
    """Encode categorical columns into numeric"""
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# Load and preprocess data
df = pd.read_csv('network_data.csv')
df_processed = preprocess(df)

# Train the model
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(df_processed)
df['anomaly'] = model.predict(df_processed)

# Save model and preprocessed column order
joblib.dump(model, 'anomaly_model.pkl')
joblib.dump(df_processed.columns.tolist(), 'model_columns.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    uploaded_file = request.files['file']
    new_data = pd.read_csv(uploaded_file)
    new_data_processed = preprocess(new_data)

    # Ensure same columns are present
    model_columns = joblib.load('model_columns.pkl')
    new_data_processed = new_data_processed.reindex(columns=model_columns, fill_value=0)

    model = joblib.load('anomaly_model.pkl')
    new_data['anomaly'] = model.predict(new_data_processed)

    anomalies = new_data[new_data['anomaly'] == -1]
    result = {
        'total_records': len(new_data),
        'anomalies_detected': len(anomalies),
        'anomaly_rows': anomalies.to_dict(orient='records')
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
