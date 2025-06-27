import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('house_data.csv')

# Encode zipcode
zipcode_encoder = LabelEncoder()
df['zipcode_encoded'] = zipcode_encoder.fit_transform(df['zipcode'])

# Save the encoder
joblib.dump(zipcode_encoder, 'zipcode_encoder.pkl')

# Prepare features and target
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors',
            'waterfront', 'view', 'grade', 'zipcode_encoded']
X = df[features]
y = df['price']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'house_model.pkl')

print("✅ Model and encoder saved successfully.")
