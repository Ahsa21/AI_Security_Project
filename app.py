from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)


model = joblib.load("original_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        data = np.array(data).reshape(1, -1)

        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]

        return jsonify({
            "prediction": int(prediction),
            "probabilities": probabilities.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True)