from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)

request_count = 0

model = joblib.load("original_model.pkl")

@app.route("/predict", methods=["POST"])

def predict():
    #global request_count
    #request_count += 1
    #if request_count > 1000:
    #   return jsonify({"error": "Rate limit exceeded"}), 429
    try:
        data = request.json["features"]
        data = np.array(data).reshape(1, -1)

        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]

        noise = np.random.normal(0, 0.15, size=probabilities.shape)
        noisy_probs = probabilities
        noisy_probs = np.clip(noisy_probs, 0, 1)
        noisy_probs = noisy_probs / noisy_probs.sum()

        return jsonify({
            "prediction": int(prediction),
            "probabilities": noisy_probs.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True)