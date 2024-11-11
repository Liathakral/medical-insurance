

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load your model (assuming you've saved it as 'model.pkl' in your Jupyter Notebook)
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Route for the home page with the form
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling form submissions and predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    age = int(request.form['age'])
    sex = 1 if request.form['sex'] == 'male' else 0
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = 1 if request.form['smoker'] == 'yes' else 0
    region = request.form['region']
    
    # Convert region into numerical value
    region_dict = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region = region_dict.get(region.lower(), -1)
    
    # Prepare the feature array for prediction
    features = np.array([[age, sex, bmi, children, smoker, region]])
    
    # Predict insurance charges
    prediction = model.predict(features)
    
    return render_template('result.html', prediction=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
