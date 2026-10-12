# Bank Marketing Propensity & Campaign Decisioning

Exploratory customer-response classification on the **UCI Bank Marketing** dataset. This retrospective notebook/script compares two class-weighted models and constructs a hypothetical threshold/value table. **The same test partition selects the preferred model and threshold; there is no untouched final policy evaluation, calibrated subscription probability or verified future-campaign gain.**

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
6. Evaluated accuracy, precision, recall, F1 and ROC-AUC **on the same held-out partition later reused for model choice**.
7. Explored score thresholds using that same test partition, not an independently validated operating policy. Class weighting means output scores are **not established as calibrated subscription probabilities**.
8. Applied arbitrary CONTACT_COST=1 and SUBSCRIPTION_VALUE=8 illustrative units to choose a test-selected threshold; no observed business value or independent policy test is available.

## Verified results

| Metric | Result |
|---|---:|
| Model with highest ROC-AUC on reused test partition | **Logistic Regression** |
| Exploratory reused-test ROC-AUC for chosen model | **0.755** |
| Precision at 0.50 | **0.267** |
| Recall at 0.50 | **0.599** |
| F1 at 0.50 | **0.369** |
| Reused-test, assumed-value maximizing threshold | **0.55** |

## Key modeling decision

The strongest methodological choice in the project was excluding **call duration** from the predictor set. Because duration is only known after the call finishes, using it for a pre-call targeting model would make test performance unrealistically optimistic.

## Business interpretation

- Accuracy alone is not an adequate metric because only 11.7% of customers subscribe.
- Threshold selection should depend on the cost of contacting a customer versus the expected value of a successful subscription.
- Logistic regression scored higher on the **reused test partition**; selecting the winner on those same outcomes precludes treating its 0.755 AUC as an independent, post-selection performance estimate.
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
2. Current `analysis.py` imports `IPython.display`; install `ipython` separately (not currently declared in `requirements.txt`) and run `python analysis.py` with access to the live UCI data source. A self-contained standard-CLI output path is **not yet verified**. Do not report reproduction as independently completed.
3. Compare model metrics and inspect the threshold-analysis tables.

## Limitations

- **Selection leakage into evaluation:** the model is chosen using test ROC-AUC; the same test is reused to optimize the illustrative economic threshold. Reported model and threshold values are exploratory and subject to post-selection optimism; a separate validation/test protocol would be needed for independent estimates.
- The campaign-value calculation uses assumed unit costs/payoffs and historical observed subscriptions. It does not establish incremental conversions, profitability or effects of contacting customers.
- The classifiers use balanced class weights; no probability calibration analysis was performed. Treat scores as rankings, not validated purchase likelihoods.
- Predictive coefficients and feature importance do not establish causal changes; time/cohort/drift robustness has not been evaluated.
- The CLI dependency/formatting defect is recorded as an open reproducibility issue, not claimed fixed by this documentation.
