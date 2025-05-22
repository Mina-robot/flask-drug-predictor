from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoders
with open('labelencoder_gender.pkl', 'rb') as f:
    labelencoder_gender = pickle.load(f)

with open('labelencoder_bp.pkl', 'rb') as f:
    labelencoder_bp = pickle.load(f)

with open('labelencoder_cholesterol.pkl', 'rb') as f:
    labelencoder_cholesterol = pickle.load(f)

with open('labelencoder_drug.pkl', 'rb') as f:
    labelencoder_drug = pickle.load(f)

# Flask app
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        age = int(data['Age'])
        sex = labelencoder_gender.transform([data['Sex']])[0]
        bp = labelencoder_bp.transform([data['BP']])[0]
        chol = labelencoder_cholesterol.transform([data['Cholesterol']])[0]
        na_to_k = float(data['Na_to_K'])
        glucose = int(data['Glucose'])

        features = np.array([[age, sex, bp, chol, na_to_k, glucose]])
        prediction = model.predict(features)[0]

        predicted_drug = labelencoder_drug.inverse_transform([prediction])[0]

        return jsonify({'predicted_drug': predicted_drug})

        # return jsonify({'predicted_drug': str(prediction)})


    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
