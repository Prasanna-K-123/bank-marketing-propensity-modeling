import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

pd.set_option("display.max_columns", 60)

bank = fetch_ucirepo(id=222)
X_raw = bank.data.features.copy()
y_raw = bank.data.targets.copy()

print("Feature shape:", X_raw.shape)
print("Target shape:", y_raw.shape)
display(X_raw.head())
display(y_raw.head())

# Canonical binary target
target_col = y_raw.columns[0]
y = (
    y_raw[target_col]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)
if y.isna().any():
    raise ValueError("Unexpected target values: " + str(y_raw[target_col].unique()))

X = X_raw.copy()
print("Positive rate:", y.mean())

# Remove duration because it is only fully known after a call ends.
leakage_cols = [c for c in X.columns if c.lower() == "duration"]
X_model = X.drop(columns=leakage_cols, errors="ignore").copy()
print("Removed leakage-prone columns:", leakage_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

logit = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

tree = Pipeline([
    ("prep", preprocess),
    ("model", DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    )),
])

models = {"Logistic Regression": logit, "Decision Tree": tree}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    })

model_results = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
display(model_results)

best_name = model_results.iloc[0]["model"]
best_model = models[best_name]
test_prob = best_model.predict_proba(X_test)[:, 1]

threshold_rows = []
for threshold in np.arange(0.10, 0.91, 0.05):
    pred_t = (test_prob >= threshold).astype(int)
    threshold_rows.append({
        "threshold": round(float(threshold), 2),
        "precision": precision_score(y_test, pred_t, zero_division=0),
        "recall": recall_score(y_test, pred_t, zero_division=0),
        "f1": f1_score(y_test, pred_t, zero_division=0),
        "predicted_positive_rate": pred_t.mean(),
    })

threshold_table = pd.DataFrame(threshold_rows)
display(threshold_table)

# Illustrative campaign economics, not the bank's actual economics.
CONTACT_COST = 1.0
SUBSCRIPTION_VALUE = 8.0
value_rows = []
for threshold in np.arange(0.05, 0.96, 0.05):
    pred_t = (test_prob >= threshold).astype(int)
    tp = int(((pred_t == 1) & (y_test.to_numpy() == 1)).sum())
    fp = int(((pred_t == 1) & (y_test.to_numpy() == 0)).sum())
    contacts = tp + fp
    net_value = tp * SUBSCRIPTION_VALUE - contacts * CONTACT_COST
    value_rows.append({
        "threshold": round(float(threshold), 2),
        "contacts": contacts,
        "true_subscribers_reached": tp,
        "net_value_units": net_value,
    })

value_table = pd.DataFrame(value_rows).sort_values("net_value_units", ascending=False)
display(value_table.head(10))
best_threshold = float(value_table.iloc[0]["threshold"])

feature_names = best_model.named_steps["prep"].get_feature_names_out()
estimator = best_model.named_steps["model"]
if hasattr(estimator, "coef_"):
    importance = pd.DataFrame({
        "feature": feature_names,
        "value": estimator.coef_[0],
    })
    importance["abs_value"] = importance["value"].abs()
    importance = importance.sort_values("abs_value", ascending=False)
    display(importance.head(15)[["feature", "value"]])
else:
    importance = pd.DataFrame({
        "feature": feature_names,
        "value": estimator.feature_importances_,
    }).sort_values("value", ascending=False)
    display(importance.head(15))

winner = model_results.iloc[0]
print("=== VERIFIED PROJECT FACTS ===")
print(f"Rows modeled: {len(X_model):,}")
print(f"Features before encoding: {X_model.shape[1]}")
print(f"Positive subscription rate: {y.mean()*100:.2f}%")
print(f"Best model by ROC-AUC: {winner['model']}")
print(f"Test ROC-AUC: {winner['roc_auc']:.3f}")
print(f"Test precision @ 0.50: {winner['precision']:.3f}")
print(f"Test recall @ 0.50: {winner['recall']:.3f}")
print(f"Test F1 @ 0.50: {winner['f1']:.3f}")
print(f"Best illustrative campaign threshold: {best_threshold:.2f}")
print("Leakage control: excluded call duration for pre-call targeting.")
