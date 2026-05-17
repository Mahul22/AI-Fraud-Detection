import shap
import joblib
import pandas as pd

model = joblib.load("fraud_model.pkl")

df = pd.read_csv("../data/creditcard.csv")

X = df.drop("Class", axis=1)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X[:100])

shap.summary_plot(shap_values, X[:100])