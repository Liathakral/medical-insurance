from flask import Flask, render_template, request
from model import predict_insurance

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Extract form data
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = int(request.form["smoker"])
        region = int(request.form["region"])
        
        # Prepare features for prediction
        features = [age, sex, bmi, children, smoker, region]
        
        # Get prediction
        prediction = predict_insurance(features)
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
