from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv(r"C:\Users\karna\fsdai\creditcard\creditcard (1).csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Get features (list)
    features = data["features"]

    # Convert to DataFrame with correct shape
    input_data = pd.DataFrame([features])

    # If your model expects full 30 features, you MUST send all
    # Temporary fix: pad with zeros
    if input_data.shape[1] < X.shape[1]:
        for i in range(X.shape[1] - input_data.shape[1]):
            input_data[i + input_data.shape[1]] = 0

    input_data = input_data.iloc[:, :X.shape[1]]

    # Predict
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "Fraud Transaction"
    else:
        result = "Normal Transaction"

    return jsonify({"prediction": result})
if_name_=="_main_":
    app.run(debug=True)