# hotel.py (balanced + curves + threshold + reliable SHAP)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import shap

st.set_page_config(page_title="LOS Risk Predictor", layout="wide")
st.title("Prolonged Length of Stay (LOS) Predictor")
st.caption("Upload your dataset, train models, compare performance, and see SHAP explanations.")
st.markdown("**Target:** Prolonged stay = 1 if `(Discharge Date - Date of Admission) ≥ 4 days`, else 0.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

example_cols = [
    "Name","Age","Gender","Blood Type","Medical Condition","Date of Admission",
    "Doctor","Hospital","Insurance Provider","Billing Amount","Room Number",
    "Admission Type","Discharge Date","Medication","Test Results"
]
with st.expander("Expected / Typical Columns"):
    st.code(", ".join(example_cols))

def _to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def build_preprocessor(cat_cols, num_cols):
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)  # sklearn >=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)         # older sklearn
    return ColumnTransformer([
        ("cat", ohe, cat_cols),
        ("num", "passthrough", num_cols),
    ])

def make_features(raw: pd.DataFrame):
    df = raw.copy()
    df["Date of Admission"] = _to_datetime(df["Date of Admission"])
    df["Discharge Date"]    = _to_datetime(df["Discharge Date"])
    df = df.dropna(subset=["Date of Admission", "Discharge Date"])

    df["LOS_days"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
    df = df[df["LOS_days"].notna() & (df["LOS_days"] >= 0)]

    df["ProlongedStay"] = (df["LOS_days"] >= 4).astype(int)

    df["Adm_DOW"]   = df["Date of Admission"].dt.dayofweek
    df["Adm_Month"] = df["Date of Admission"].dt.month
    df["Adm_Hour"]  = df["Date of Admission"].dt.hour

    base_feats = [
        "Age","Gender","Blood Type","Medical Condition","Admission Type",
        "Insurance Provider","Adm_DOW","Adm_Month","Adm_Hour"
    ]
    feature_cols = [c for c in base_feats if c in df.columns]

    X = df[feature_cols].copy()
    y = df["ProlongedStay"]

    cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown")

    pre = build_preprocessor(cat_cols, num_cols)
    return df, X, y, pre, cat_cols, num_cols, feature_cols

if uploaded is None:
    st.info("Upload your CSV to continue.")
    st.stop()

df_raw = pd.read_csv(uploaded)
df_raw.columns = [c.strip() for c in df_raw.columns]

missing = [c for c in example_cols if c not in df_raw.columns]
if missing:
    st.warning(f"Some expected columns are missing: {missing}. Proceeding with available columns.")

df, X, y, pre, cat_cols, num_cols, feature_cols = make_features(df_raw)

st.subheader("Data Preview")
st.dataframe(df.head(20))

if len(df) == 0:
    st.error("No valid rows after date cleaning. Check your date formats.")
    st.stop()
if y.nunique() < 2:
    st.error("Target has a single class. You need both short and prolonged stays.")
    st.stop()

# Class balance view
c0, c1 = (y == 0).sum(), (y == 1).sum()
st.info(f"Class balance -> Short stays (0): {c0}, Prolonged (1): {c1}")

# Optional downsample for interactive speed
MAX_ROWS = 20000
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)
    X = X.loc[df.index]
    y = y.loc[df.index]

# -------- Modeling (with class weights for imbalance) --------
models = {
    "LogisticRegression (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest (balanced)": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced_subsample"
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),  # no class_weight here
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

results, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    # store both the fitted pipe and raw clf for SHAP later
    fitted[name] = pipe
    # probs for AUC
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")
    # default threshold 0.5 for the table
    y_pred_05 = (y_prob >= 0.5).astype(int) if y_prob is not None else pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred_05)
    f1  = f1_score(y_test, y_pred_05)
    results.append((name, roc, acc, f1))

res_df = pd.DataFrame(results, columns=["Model","ROC_AUC","Accuracy@0.5","F1@0.5"]).sort_values("ROC_AUC", ascending=False)
st.subheader("Model Comparison")
st.dataframe(res_df.style.format({"ROC_AUC":"{:.3f}","Accuracy@0.5":"{:.3f}","F1@0.5":"{:.3f}"}))

best_name = res_df.iloc[0]["Model"]
best_pipe = fitted[best_name]
st.success(f"Best model: {best_name}")

# -------- Curves + threshold tuning --------
st.subheader("Threshold & Curves")

if hasattr(best_pipe.named_steps["clf"], "predict_proba"):
    y_prob_best = best_pipe.predict_proba(X_test)[:, 1]
else:
    # fall back to decision function if available
    if hasattr(best_pipe.named_steps["clf"], "decision_function"):
        z = best_pipe.named_steps["clf"].decision_function(X_test)
        # map to 0-1 range
        y_prob_best = (z - z.min()) / (z.max() - z.min() + 1e-9)
    else:
        y_prob_best = None

if y_prob_best is not None:
    # Threshold slider
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    y_pred_thr = (y_prob_best >= thr).astype(int)

    cm = confusion_matrix(y_test, y_pred_thr)
    st.write("**Confusion Matrix (at selected threshold)**")
    st.write(cm)

    st.write("**Classification Report (at selected threshold)**")
    st.text(classification_report(y_test, y_pred_thr, digits=3))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    ap = average_precision_score(y_test, y_prob_best)
    prec, rec, _ = precision_recall_curve(y_test, y_prob_best)

    fig1 = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    st.pyplot(fig1)

    fig2 = plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
    st.pyplot(fig2)
else:
    st.info("Best model doesn’t expose predict_proba; curves and threshold slider are skipped.")

# -------- SHAP Explainability --------
st.subheader("Explainability (SHAP)")

pre_fitted = best_pipe.named_steps["pre"]
clf = best_pipe.named_steps["clf"]

# Transform test data
X_te_trans = pre_fitted.transform(X_test)
sample_te = min(300, X_te_trans.shape[0])

# Feature names
feature_names = []
if len(pre_fitted.transformers_) > 0:
    # cat
    try:
        ohe = pre_fitted.named_transformers_["cat"]
        feature_names.extend(ohe.get_feature_names_out([*ohe.feature_names_in_]).tolist())
    except Exception:
        # fallback when feature names are not exposed the same way
        if hasattr(pre_fitted.named_transformers_.get("cat", None), "get_feature_names_out"):
            feature_names.extend(pre_fitted.named_transformers_["cat"].get_feature_names_out().tolist())
# num
feature_names.extend([c for c in X.columns if c not in feature_names])

# Convert to dense for plotting a small sample
if hasattr(X_te_trans, "toarray"):
    X_te_small = X_te_trans[:sample_te].toarray()
else:
    X_te_small = np.array(X_te_trans[:sample_te])

def shap_summary(values, X_small, names):
    fig = plt.figure()
    shap.summary_plot(values, X_small, feature_names=names, show=False)
    st.pyplot(fig)

is_tree = any(k in best_name for k in ["RandomForest","GradientBoosting"])
try:
    if is_tree:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_te_small)
        st.caption("Global importance (top drivers of prolonged stay):")
        values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_summary(values_for_plot, X_te_small, feature_names)

        st.markdown("**Local explanation for one case:**")
        idx = st.number_input("Row index (0-based) for SHAP detail", min_value=0, max_value=sample_te-1, value=0, step=1)
        fig2 = plt.figure()
        sv = values_for_plot[idx]
        shap.bar_plot(sv, feature_names=feature_names, max_display=15, show=False)
        st.pyplot(fig2)
    else:
        st.info("Best model is not tree-based; using slower KernelExplainer on a small sample.")
        # small background
        bg_idx = np.random.choice(X_train.index, size=min(200, len(X_train)), replace=False)
        X_bg = pre_fitted.transform(X_train.loc[bg_idx])
        if hasattr(X_bg, "toarray"):
            X_bg = X_bg.toarray()
        def model_fn(z):
            return clf.predict_proba(z)[:, 1]
        explainer = shap.KernelExplainer(model_fn, X_bg, link="logit")
        shap_values = explainer.shap_values(X_te_small, nsamples=100)
        st.caption("Global importance (top drivers of prolonged stay):")
        shap_summary(shap_values, X_te_small, feature_names)
except Exception as e:
    st.warning(f"SHAP explanation skipped: {e}")

# -------- Batch scoring --------
st.subheader("Batch Scoring (simulate new admissions)")
n = st.slider("How many random rows to score from your file?", min_value=5, max_value=50, value=10, step=5)
sample_rows = df.sample(n=n, random_state=1)

to_score = sample_rows[feature_cols]
p = best_pipe.predict_proba(to_score)[:, 1] if hasattr(best_pipe.named_steps["clf"], "predict_proba") else None
if p is None:
    st.info("Best model has no predict_proba; batch scoring uses class labels only.")
    pred_label = best_pipe.predict(to_score)
    pred_prob = np.full_like(pred_label, np.nan, dtype=float)
else:
    pred_prob = p
    pred_label = (pred_prob >= (thr if 'thr' in locals() else 0.5)).astype(int)

cols_safe = [c for c in ["Name","Age","Gender","Medical Condition","Admission Type","Doctor","Hospital","Insurance Provider","Room Number","Date of Admission"] if c in sample_rows.columns]
out = sample_rows[cols_safe].copy()
out["ProlongedStay_Prob"] = pred_prob
out["ProlongedStay_Pred"] = pred_label
st.dataframe(out)
st.download_button("Download predictions (CSV)", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.markdown("""
**Notes for portfolio**
- Class imbalance handled with class weights and threshold tuning.
- Curves: ROC and Precision–Recall added; use AP (average precision) for imbalanced data.
- Explainability: SHAP global & local (fast on tree models).
- Leakage avoided: excluded discharge- and post-admission information from features.
""")

