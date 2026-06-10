import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample Data
data = {
    'income': [50000, 60000, 30000, 80000, 20000, 70000, 40000],
    'debt': [10000, 5000, 15000, 2000, 18000, 4000, 12000],
    'credit_score': [1, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['income', 'debt']]
y = df['credit_score']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# New Person Prediction
new_person = [[55000, 7000]]
prediction = model.predict(new_person)

if prediction[0] == 1:
    print("Creditworthy")
else:
    print("Not Creditworthy")