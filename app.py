from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

app = Flask(__name__)

# Load the model from the file
model = joblib.load('linear.pkl')

# Define the training data for encoding
data = {
    'Outlook': ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Sunny', 'Rainy', 'Overcast', 'Overcast', 'Sunny'],
    'Temp': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'Play Golf': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical features
le_out = LabelEncoder()
le_temp = LabelEncoder()
le_hum = LabelEncoder()
le_windy = LabelEncoder()
le_play = LabelEncoder()

df['le_out'] = le_out.fit_transform(df['Outlook'])
df['le_temp'] = le_temp.fit_transform(df['Temp'])
df['le_hum'] = le_hum.fit_transform(df['Humidity'])
df['le_windy'] = le_windy.fit_transform(df['Windy'])
df['Play Golf'] = le_play.fit_transform(df['Play Golf'])

inputs_n = df[['le_out', 'le_temp', 'le_hum', 'le_windy']]
target_n = df['Play Golf']

# Define the LabelEncoders for use in prediction
# Ensure the encoders are saved and loaded correctly if needed
# For simplicity, we assume they are loaded in the same way as the model

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    outlook = request.form['outlook']
    temp = request.form['temp']
    humidity = request.form['humidity']
    windy = request.form['windy']

    # Encode input data
    outlook_encoded = le_out.transform([outlook])[0]
    temp_encoded = le_temp.transform([temp])[0]
    humidity_encoded = le_hum.transform([humidity])[0]
    windy_encoded = le_windy.transform([windy])[0]

    # Make prediction
    prediction_encoded = model.predict([[outlook_encoded, temp_encoded, humidity_encoded, windy_encoded]])[0]
    prediction = le_play.inverse_transform([prediction_encoded])[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
