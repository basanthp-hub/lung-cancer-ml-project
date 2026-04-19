from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Lung Cancer Prediction App Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Dummy logic (for demo)
    age = data.get("age", 0)
    
    if age > 50:
        result = "High Risk"
    else:
        result = "Low Risk"
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)