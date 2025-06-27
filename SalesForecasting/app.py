from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Load and prepare the data
df = pd.read_csv('sales_data.csv')

# Convert 'data' column to datetime
df['data'] = pd.to_datetime(df['data'])

# Create a numerical month index 
df['Month_num'] = df['data'].dt.month + 12 * (df['data'].dt.year - df['data'].dt.year.min())

# Train the model using 'Month_num' and 'venda'
X = df[['Month_num']]
y = df['venda']
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'forecast_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    
    # Get months and start_date from frontend
    months_ahead = int(data['months'])
    start_date = pd.to_datetime(data['start_date'])

    # Reload the model
    model = joblib.load('forecast_model.pkl')

    # Recalculate start_month_num from selected date
    start_month_num = start_date.month + 12 * (start_date.year - df['data'].dt.year.min())

    # Future month numbers to predict
    future_months = [[start_month_num + i] for i in range(1, months_ahead + 1)]
    predictions = model.predict(future_months)

    # Generate future dates based on start_date
    forecast_dates = [(start_date + pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(1, months_ahead + 1)]

    # Return prediction as JSON
    return jsonify(dict(zip(forecast_dates, predictions.round(2).tolist())))

if __name__ == '__main__':
    app.run(debug=True)
