from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Define paths
MODEL_DIR = '.'
REG_MODEL_PATH = os.path.join(MODEL_DIR, 'aqi_model.pkl')
CLUSTER_MODEL_PATH = os.path.join(MODEL_DIR, 'cluster_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
CLUSTER_MAP_PATH = os.path.join(MODEL_DIR, 'cluster_map.pkl')

# Define features
features = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']

# Function to train and save models once
def train_models():
    df = pd.read_csv('air_quality.csv')
    df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')
    df = df.dropna(subset=features + ['aqi'])

    X = df[features]
    y = df['aqi']

    # Regression
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    joblib.dump(reg_model, REG_MODEL_PATH)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    # Clustering
    cluster_model = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = cluster_model.fit_predict(X_scaled)
    joblib.dump(cluster_model, CLUSTER_MODEL_PATH)

    # Cluster map based on AQI average
    cluster_means = df.groupby('cluster')['aqi'].mean().sort_values()
    cluster_map = {cluster_id: label for cluster_id, label in zip(cluster_means.index, ['Low', 'Moderate', 'High'])}
    joblib.dump(cluster_map, CLUSTER_MAP_PATH)

# Train only once 
if not all(os.path.exists(p) for p in [REG_MODEL_PATH, CLUSTER_MODEL_PATH, SCALER_PATH, CLUSTER_MAP_PATH]):
    train_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[f]) for f in features]

    # Load models
    reg_model = joblib.load(REG_MODEL_PATH)
    cluster_model = joblib.load(CLUSTER_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cluster_map = joblib.load(CLUSTER_MAP_PATH)

    # Predict AQI
    predicted_aqi = reg_model.predict([input_data])[0]

    # Predict cluster
    input_scaled = scaler.transform([input_data])
    cluster = int(cluster_model.predict(input_scaled)[0])
    pollution_level = cluster_map.get(cluster, "Unknown")

    return jsonify({
        'aqi': round(predicted_aqi, 2),
        'cluster': cluster,
        'level': pollution_level
    })

if __name__ == '__main__':
    app.run(debug=True)
