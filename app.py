# from flask import Flask,request,render_template
# import numpy as np
# import pandas
# import sklearn
# import pickle
# import pickle
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model, f)


# # importing model
# model = pickle.load(open('model.pkl','rb'))
# sc = pickle.load(open('standscaler.pkl','rb'))
# ms = pickle.load(open('minmaxscaler.pkl','rb'))

# # creating flask app
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route("/predict",methods=['POST'])
# def predict():
#     N = request.form['Nitrogen']
#     P = request.form['Phosporus']
#     K = request.form['Potassium']
#     temp = request.form['Temperature']
#     humidity = request.form['Humidity']
#     ph = request.form['Ph']
#     rainfall = request.form['Rainfall']

#     feature_list = [N, P, K, temp, humidity, ph, rainfall]
#     single_pred = np.array(feature_list).reshape(1, -1)

#     scaled_features = ms.transform(single_pred)
#     final_features = sc.transform(scaled_features)
#     prediction = model.predict(final_features)

#     crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
#                  8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#                  14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#                  19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

#     if prediction[0] in crop_dict:
#         crop = crop_dict[prediction[0]]
#         result = "{} is the best crop to be cultivated right there".format(crop)
#     else:
#         result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
#     return render_template('index.html',result = result)




# # python main
# if __name__ == "__main__":
#     app.run(debug=True)


# (2nd)
# from flask import Flask, request, render_template
# import numpy as np
# import pandas
# import sklearn
# import pickle

# # 🔴 ERROR in your code: You were saving the model again BEFORE loading it
# # ❌ Don't do this unless you're training the model here. Remove this:
# # with open('model.pkl', 'wb') as f:
# #     pickle.dump(model, f)

# # ✅ Correct: Import the model and scalers
# model = pickle.load(open('model.pkl', 'rb'))
# sc = pickle.load(open('standscaler.pkl', 'rb'))  # StandardScaler
# ms = pickle.load(open('minmaxscaler.pkl', 'rb'))  # MinMaxScaler

# # ✅ Create Flask app
# app = Flask(__name__)

# # ✅ Home route
# @app.route('/')
# def index():
#     return render_template("index.html")

# # ✅ Prediction route
# @app.route("/predict", methods=['POST'])
# def predict():
#     try:
#         N = float(request.form['Nitrogen'])
#         P = float(request.form['Phosporus'])
#         K = float(request.form['Potassium'])
#         temp = float(request.form['Temperature'])
#         humidity = float(request.form['Humidity'])
#         ph = float(request.form['Ph'])
#         rainfall = float(request.form['Rainfall'])

#         # ✅ Prepare input
#         feature_list = [N, P, K, temp, humidity, ph, rainfall]
#         single_pred = np.array(feature_list).reshape(1, -1)

#         # ✅ Apply scaling
#         scaled_features = ms.transform(single_pred)
#         final_features = sc.transform(scaled_features)

#         # ✅ Make prediction
#         prediction = model.predict(final_features)

#         # ✅ Map prediction to crop name
#         crop_dict = {
#             1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
#             8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
#             14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
#             19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
#         }

#         if prediction[0] in crop_dict:
#             crop = crop_dict[prediction[0]]
#             result = f"{crop} is the best crop to be cultivated right there."
#         else:
#             result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
#     except Exception as e:
#         result = f"Error: {str(e)}"

#     return render_template('index.html', result=result)

# # ✅ Run the app
# if __name__ == "__main__":
#     app.run(debug=True)


# # (3rd)
# from flask import Flask, render_template, request
# import pickle

# app = Flask(__name__)

# # Load the model
# model = pickle.load(open('model.pkl', 'rb'))

# # Home route
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Prediction route (if needed)
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract form values
#     N = float(request.form['Nitrogen'])
#     # P = float(request.form['Phosporus'])
#     P = float(request.form['Phosphorus'])  # Corrected spelling

#     K = float(request.form['Potassium'])
#     temperature = float(request.form['Temperature'])
#     humidity = float(request.form['Humidity'])
#     ph = float(request.form['Ph'])
#     rainfall = float(request.form['Rainfall'])

#     # Predict
#     prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
#     crop = prediction[0]

#     return render_template('index.html', prediction_text=f'Recommended Crop: {crop}')

# if __name__ == '__main__':
#     app.run(debug=True)


# # (4rth)
# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load model and scalers
# model = pickle.load(open('model.pkl', 'rb'))
# minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
# standard_scaler = pickle.load(open('standscaler.pkl', 'rb'))
# label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input values from form
#         N = float(request.form['Nitrogen'])
#         P = float(request.form['Phosphorus'])  # ✔️ Ensure spelling matches HTML form
#         K = float(request.form['Potassium'])
#         temperature = float(request.form['Temperature'])
#         humidity = float(request.form['Humidity'])
#         ph = float(request.form['Ph'])
#         rainfall = float(request.form['Rainfall'])

#         # Create feature array
#         input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

#         # Apply both scalers (same as training)
#         scaled = minmax_scaler.transform(input_features)
#         final_input = standard_scaler.transform(scaled)

#         # Predict
#         prediction = model.predict(final_input)
#         predicted_crop = label_encoder.inverse_transform(prediction)[0]  # Convert label to name

#         return render_template('index.html', prediction_text=f"🌱 Recommended Crop: {predicted_crop}")

#     except Exception as e:
#         return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

# if __name__ == '__main__':
#     app.run(debug=True)

# (5)
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
