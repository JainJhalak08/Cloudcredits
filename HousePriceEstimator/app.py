from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib

app = Flask(__name__)
app.secret_key = 'secret123'  # Required for session management

# Load trained model and zipcode encoder
model = joblib.load('house_model.pkl')
zipcode_encoder = joblib.load('zipcode_encoder.pkl')

# Load available zipcodes for dropdown
zipcodes = sorted(zipcode_encoder.classes_.tolist())

@app.route('/')
def index():
    prediction = session.pop('prediction', None)
    return render_template('index.html', prediction=prediction, zipcodes=zipcodes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'sqft_living': float(request.form['sqft_living']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': float(request.form['bathrooms']),
            'floors': float(request.form['floors']),
            'waterfront': int(request.form['waterfront']),
            'view': int(request.form['view']),
            'grade': int(request.form['grade']),
            'zipcode': request.form['zipcode']
        }

        # Encode zipcode
        input_data['zipcode_encoded'] = int(zipcode_encoder.transform([input_data['zipcode']])[0])
        del input_data['zipcode']

        input_df = pd.DataFrame([input_data])
        prediction = round(model.predict(input_df)[0], 2)

        session['prediction'] = prediction
        return redirect(url_for('index'))

    except Exception as e:
        session['prediction'] = f"Error: {str(e)}"
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
