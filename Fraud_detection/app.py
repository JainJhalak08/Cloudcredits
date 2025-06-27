from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv('fraud_data.csv')
df.columns = df.columns.str.strip()  

# Features and target based on available data
X = df[['Amount']]  
y = df['Class']    

# Train model 
if not os.path.exists('fraud_model.pkl'):
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, 'fraud_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    uploaded_file = request.files['file']
    new_data = pd.read_csv(uploaded_file)
    new_data.columns = new_data.columns.str.strip()

    if 'Amount' not in new_data.columns:
        return jsonify({'error': "Uploaded file must contain an 'Amount' column."}), 400

    model = joblib.load('fraud_model.pkl')
    predictions = model.predict(new_data[['Amount']])
    new_data['prediction'] = predictions
    frauds = new_data[new_data['prediction'] == 1]

    result = {
        'total_records': len(new_data),
        'fraudulent': len(frauds),
        'fraud_rows': frauds.to_dict(orient='records')
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
