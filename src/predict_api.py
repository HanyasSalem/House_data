# Importing necessary libraries : flask, pandas, jolib
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Assigning file paths
model_path = r"D:\House_data\models\supervised_rf.joblib"
X_train_path = r"D:\House_data\train_test\X_train.csv"

#  Loading the model grid_rf

grid_rf = joblib.load(model_path)  # GridSearchCV object - .predict() automatically uses grid_rf.best_estimator_
print ("model loaded successfully ...")

# Extracting the features for user 
feature_columns = pd.read_csv(X_train_path, nrows=0).columns.tolist()

print(feature_columns)

# Create the Flask application 
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def get_predict():
    """
    Expects a JSON body where each key is a feature name (matching
    feature_columns above) and each value is that feature's number
    for one house. Returns the predicted price as JSON.
    """
    
    data = request.get_json(force=True)

    try:
        input_row = pd.DataFrame([data])[feature_columns]
    except KeyError as missing:
        
        return jsonify({
            "error": f"a feature is missing or missnamed: {missing}",
            "expected_features": feature_columns
        }), 400

    
    predicted_price = grid_rf.predict(input_row)[0]

    return jsonify({"predicted_price": round(float(predicted_price), 2)})


if __name__ == "__main__":

    app.run(port=5000, debug=True)
