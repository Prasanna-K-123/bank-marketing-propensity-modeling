# Bank Marketing Propensity & Campaign Decisioning

Customer-response classification on the **UCI Bank Marketing** historical dataset, with two separately documented evaluations. **Registered V2** separates training, score calibration, model/threshold selection and a final 9,043-row holdout: ROC-AUC **0.7529** and Brier **0.08691**. The historical V1 script reuses its test partition for selection and remains exploratory. Neither evaluation establishes causal campaign uplift or future profitability.

## Business question

Can historical customer and campaign information be used to identify customers more likely to subscribe to a term deposit before the marketing call takes place?

## Dataset

**UCI Bank Marketing** dataset.

- Rows modeled: **45,211**
- Features before encoding: **15**
- Positive subscription rate: **11.70%**

## Historical V1 modeling workflow

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

## Historical V1 exploratory results

| Metric | Result |
|---|---:|
| Model with highest ROC-AUC on reused test partition | **Logistic Regression** |
| Exploratory reused-test ROC-AUC for chosen model | **0.755** |
| Precision at 0.50 | **0.267** |
| Recall at 0.50 | **0.599** |
| F1 at 0.50 | **0.369** |
| Reused-test, assumed-value maximizing threshold | **0.55** |

## Registered V2: independent final-holdout policy check (12 October 2026)

This is **a new, separately registered evaluation**, not a retrospective edit of V1's original test set. The [protocol](research_v2/REGISTERED_PROTOCOL.md) was committed before running the full dataset through V2. In the 45,211-row UCI sample, V2 uses disjoint random stratified partitions for training (22,605), probability calibration (6,781), model/threshold selection (6,782), and a final untouched test (9,043). The selected model is logistic regression, with calibration learned on a separate sample; the **hypothetical** contact-score threshold is **0.15**.

| Frozen V2 final-test measure | Result |
|---|---:|
| ROC-AUC | **0.7529** (1,000-bootstrap 95% interval 0.7369–0.7697) |
| Brier probability error | **0.08691** (interval 0.08235–0.09134) |
| Log loss | **0.30491** |
| Ten-bin ECE | **0.01077** |
| Contacted under hypothetical threshold | **1,390 / 9,043 (15.37%)** |
| Historical subscribers reached | **505** (precision 36.33%; recall 47.73%) |
| Arbitrary illustrative net units = 8 × reached subscribers − contacts | **+2,650** (contact-all −579; contact-none 0) |

**Economic caveat:** The last row is an arithmetic thought experiment using historical subscription outcomes and completely assumed unit payoffs. It is **not** incremental conversion, causally identified campaign uplift, realized profitability, production targeting, or a valid ROI estimate. All observed customers underwent historical contacts. Random stratification is neither temporal nor group-held-out, and calibration is only assessed on the retrospective sample. The old V1 ROC-AUC 0.755 and test-selected threshold 0.55 refer to a different evaluation scheme; they are retained but **not compared as a controlled performance gain**.

The full result, all adverse cases, provenance fingerprints and uncertainty are preserved in [registered result JSON](research_v2/results/registered_policy_v2.json) and [method/results audit](research_v2/RESULTS_AUDIT.md). [First green source-and-artifact CI run](https://github.com/Prasanna-K-123/bank-marketing-propensity-modeling/actions/runs/38296515143) and [pinned-dependency successful replay](https://github.com/Prasanna-K-123/bank-marketing-propensity-modeling/actions/runs/38297248585) are separately inspectable. An intermediate strict-equality replay failed and remains recorded; it was not silently overwritten. The frozen source release `b4317f5e83be3411c005062212957addead7212f` passed two matching-main replays: [38297965756](https://github.com/Prasanna-K-123/bank-marketing-propensity-modeling/actions/runs/38297965756) and [38298073331](https://github.com/Prasanna-K-123/bank-marketing-propensity-modeling/actions/runs/38298073331). These verify V2 reproduction; they do not validate future deployment.

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

## Reproducing the evaluations

For registered V2, from the repository root:

```bash
python -m pip install -r research_v2/requirements.txt
python -m pytest research_v2/test_registered_policy.py -q
python research_v2/validate_policy.py
```

The validator checks the fixed dataset/protocol fingerprints, selected model/threshold and report structure/counts. Only finite floating diagnostics permit the documented 1e-12 replay tolerance; the frozen result is not silently replaced.

### Historical V1

1. Install the packages in `requirements.txt`.
2. Current `analysis.py` imports `IPython.display`; `ipython` has now been added to `requirements.txt`. Run `python analysis.py` with access to the live UCI data source. The dependency declaration and import have been read back, but a complete fresh environment install and end-to-end source download have **not** been independently replayed; do not treat this as a certified reproducibility pass. Do not report reproduction as independently completed.
3. Compare model metrics and inspect the threshold-analysis tables.

## Limitations

- **Historical V1 selection leakage:** the original model was selected on its test ROC-AUC and its hypothetical threshold optimized on the same partition. V1's 0.755 AUC and 0.55 threshold remain exploratory and cannot be used as untouched evidence. The separately registered V2 now provides an independent retrospective final test, but **still not temporal/generalization or causal policy validation**.
- The campaign-value calculation uses assumed unit costs/payoffs and historical observed subscriptions. It does not establish incremental conversions, profitability or effects of contacting customers.
- Historical V1 uses balanced class weights without probability calibration. V2 fits sigmoid calibration on its separate calibration partition and reports final-test probability diagnostics; this does not establish future or subgroup calibration.
- Predictive coefficients and feature importance do not establish causal changes; time/cohort/drift robustness has not been evaluated.
- The historical V1 fresh-environment, live-download CLI check remains unverified. The registered V2 source/data/report replay is independently verified by the linked CI runs.
