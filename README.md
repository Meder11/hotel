# hotel
I built a small app that predicts prolonged hospital length of stay at admission and explains the “why.”
What it does

Computes LOS from admission and discharge dates, then predicts if a patient will stay ≥ 4 days
Uses Logistic Regression, Random Forest, and Gradient Boosting and compares results
Handles class imbalance and lets you adjust the decision threshold
Uses SHAP to show global and per-patient feature impact in plain English
Why it matters
Better early signals help with bed planning, staffing, and discharge coordination.
Tech
Python, scikit-learn, Streamlit, SHAP, pandas
