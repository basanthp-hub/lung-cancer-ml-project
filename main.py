from flask import Flask, request, jsonify
from model import train_model

app = Flask(__name__)

# Load ML model
model = train_model()

@app.route("/")
def home():
    return "Iris ML Prediction App Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Get 4 features (Iris dataset needs 4 values)
    features = [
        data.get("f1", 0),
        data.get("f2", 0),
        data.get("f3", 0),
        data.get("f4", 0)
    ]

    prediction = model.predict([features])

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)