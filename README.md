# Bank Marketing Propensity & Campaign Decisioning

Predictive customer-response modeling using the **UCI Bank Marketing** dataset. The project compares interpretable classification models, handles class imbalance, controls target leakage, and links model thresholds to campaign decision-making.

## Business question

Can historical customer and campaign information be used to identify customers more likely to subscribe to a term deposit before the marketing call takes place?

## Dataset

**UCI Bank Marketing** dataset.

- Rows modeled: **45,211**
- Features before encoding: **15**
- Positive subscription rate: **11.70%**

## Modeling workflow

1. Loaded the UCI dataset and audited the target distribution.
2. Removed `duration` from the feature set because it is only fully known after a call ends and would create leakage in a pre-call targeting use case.
3. Used a stratified train/test split to preserve the imbalanced class distribution.
4. Built preprocessing pipelines for numeric and categorical variables.
5. Compared:
   - Logistic Regression with balanced class weights
   - Decision Tree with depth and leaf-size controls
6. Evaluated Accuracy, Precision, Recall, F1 and ROC-AUC.
7. Tested probability thresholds rather than assuming 0.50 is always the right operating point.
8. Added an illustrative cost-benefit scenario to show how a business threshold could be selected.

## Verified results

| Metric | Result |
|---|---:|
| Best model by ROC-AUC | **Logistic Regression** |
| Test ROC-AUC | **0.755** |
| Precision at 0.50 | **0.267** |
| Recall at 0.50 | **0.599** |
| F1 at 0.50 | **0.369** |
| Illustrative value-maximizing threshold | **0.55** |

## Key modeling decision

The strongest methodological choice in the project was excluding **call duration** from the predictor set. Because duration is only known after the call finishes, using it for a pre-call targeting model would make test performance unrealistically optimistic.

## Business interpretation

- Accuracy alone is not an adequate metric because only 11.7% of customers subscribe.
- Threshold selection should depend on the cost of contacting a customer versus the expected value of a successful subscription.
- Logistic Regression generalized better than the constrained Decision Tree on ROC-AUC in this implementation.
- The project treats model outputs as predictive associations, not causal evidence that changing a feature would change subscription behavior.

## Technology stack

- Python
- Pandas, NumPy, Matplotlib
- scikit-learn
- Logistic Regression
- Decision Tree
- One-Hot Encoding and Standardization
- ROC-AUC, Precision, Recall, F1

## Repository structure

```text
bank-marketing-propensity-modeling/
├── README.md
├── analysis.py
└── requirements.txt
```

## Reproducing the analysis

1. Install the packages in `requirements.txt`.
2. Run `analysis.py` in an IPython/Jupyter environment or adapt the `display()` calls to `print()` for a standard Python shell.
3. Compare model metrics and inspect the threshold-analysis tables.

## Limitations

- The campaign-value calculation is an illustrative scenario rather than the bank's real economics.
- Predictive coefficients and feature importance do not establish causality.
- Performance may vary under different time periods, populations, or campaign strategies.
