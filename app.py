from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

MODEL_PATHS = {
    "LinearRegression": "insurance_model_LinearRegression.pkl",
    "RandomForest": "insurance_model_rf.pkl",
    "SVR": "insurance_model_svr.pkl"
}

def load_model(model_name):
    model_path = MODEL_PATHS.get(model_name)
    if model_path:
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
            if model_name in ["SVR", "RandomForest"] and isinstance(loaded_data, dict):
                return loaded_data['model'], loaded_data['scaler_Y']
            else:
                return loaded_data, None
    return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])
    
    features = np.array([[age, sex, bmi, children, smoker, region]])
    
    model, scaler_Y = load_model(model_name)
    if model is not None:
        if scaler_Y:  
            prediction = model.predict(features)
            prediction = scaler_Y.inverse_transform(prediction.reshape(-1, 1))  
            prediction_value = prediction[0][0]  
        else:
            prediction = model.predict(features)  
            prediction_value = prediction[0]  

        return render_template('result.html', prediction=round(prediction_value, 2))
    else:
        return "Model not found", 400


if __name__ == '__main__':
    app.run(debug=True)
