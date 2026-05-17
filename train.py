import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

# Load the dataset
df = pd.read_csv('C:/Users/mahul/Desktop/Socket Programming/Project/data/creditcard.csv')
X=df.drop('Class',axis=1)
y=df["Class"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,scale_pos_weight=577,eval_metric='logloss')

# Train the model
model.fit(X_train,y_train)

#Predictions
preds=model.predict(X_test)

#Probabilitites
probabs=model.predict_proba(X_test)[:,1]

# Metrics
print(classification_report(y_test, preds))

roc = roc_auc_score(y_test, probabs)

print(f"ROC-AUC Score: {roc}")

# Save model
joblib.dump(model, "fraud_model.pkl")

print("Model saved successfully.")

