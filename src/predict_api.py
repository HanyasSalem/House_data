# predict_api.py
# ------------------------------------------------------------------
# Deploys the trained house-price Random Forest pipeline as a local
# web API using Flask. Postman (or any HTTP client) can send a house's
# feature values to this API and get back a predicted price.
#
# How to run:
#   1. Make sure Flask is installed:  pip install flask
#   2. Run this file:  python predict_api.py
#   3. Leave the terminal window open - it IS the running server.
# ------------------------------------------------------------------

from flask import Flask, request, jsonify
import pandas as pd
import joblib

# ---- File paths (matches your existing predict.py setup) ----
MODEL_PATH = r"D:\House_data\models\supervised_rf.joblib"
X_TRAIN_PATH = r"D:\House_data\train_test\X_train.csv"

# ---- Load the model and the expected feature schema ONCE at startup ----
# Loading here (not inside get_predict) matters: loading a model from
# disk is slow, and we don't want to redo it on every single API call.
# The API will stay fast because the model is already in memory.
print("Loading trained pipeline...")
grid_rf = joblib.load(MODEL_PATH)  # GridSearchCV object - .predict() automatically uses grid_rf.best_estimator_

# We only need the column NAMES from X_train, not the actual data,
# so nrows=0 reads just the header row - fast even on a large file.
feature_columns = pd.read_csv(X_TRAIN_PATH, nrows=0).columns.tolist()

print(f"Model loaded successfully. This API expects {len(feature_columns)} features:")
print(feature_columns)

# ---- Create the Flask application ----
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def get_predict():
    """
    Expects a JSON body where each key is a feature name (matching
    feature_columns above) and each value is that feature's number
    for one house. Returns the predicted price as JSON.
    """
    # 1. Parse the JSON body sent by Postman
    data = request.get_json(force=True)

    # 2. Arrange the incoming values into a single-row table, in the
    #    exact column order the model was trained on. Using the saved
    #    column list (instead of trusting request order) prevents the
    #    silent "right numbers, wrong column" bug tree models are
    #    vulnerable to.
    try:
        input_row = pd.DataFrame([data])[feature_columns]
    except KeyError as missing:
        # A friendly error instead of a server crash if a feature is
        # missing or misspelled in the Postman request body.
        return jsonify({
            "error": f"Missing or misnamed feature(s): {missing}",
            "expected_features": feature_columns
        }), 400

    # 3. Predict the price
    predicted_price = grid_rf.predict(input_row)[0]

    # 4. Send the result back as JSON. round() and float() keep the
    #    output clean and JSON-serializable (numpy numbers aren't).
    return jsonify({"predicted_price": round(float(predicted_price), 2)})


if __name__ == "__main__":
    # debug=True auto-reloads the server when you save changes to this
    # file - handy while testing, but turn it off for a real client demo.
    app.run(port=5000, debug=True)
