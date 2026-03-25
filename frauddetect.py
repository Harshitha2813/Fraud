import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\karna\fsdai\creditcard\creditcard (1).csv")

# Inputs and outputs
X = df.drop("Class", axis=1)
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# ---- Prediction Example ----

sample_transaction = X_test.iloc[0].values.reshape(1,-1)

prediction = model.predict(sample_transaction)

if prediction[0] == 1:
    print("Fraudulent Transaction Detected")

else:
    print("Normal Transaction")

