from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scalers
model = pickle.load(open("model.pkl", "rb"))
minmax_scaler = pickle.load(open("minmaxscaler.pkl", "rb"))
standard_scaler = pickle.load(open("standscaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Arrange input and scale
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        minmax_scaled = minmax_scaler.transform(input_data)
        final_scaled = standard_scaler.transform(minmax_scaled)

        # Predict
        prediction_encoded = model.predict(final_scaled)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        result = f"{prediction_label} is the best crop to be cultivated right there."
    except Exception as e:
        result = f"Error in prediction: {e}"

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
