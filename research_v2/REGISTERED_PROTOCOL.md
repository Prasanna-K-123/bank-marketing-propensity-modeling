# Registered Bank Marketing Policy-Selection Validation (V2)

This registration is prospective relative to the V2 test; V1's 0.755 AUC and 0.55 threshold remain historical and do not select V2 outcomes.

## Input contract
UCI Bank Marketing full sample, dataset ID 222, target yes/no converted to 1/0, exactly 45,211 rows. Exclude call `duration` from the pre-call feature set. Record source dataset shape, label count and canonical data SHA256. Fail closed on missing duration, unexpected labels or sample size; never swap datasets silently.

## Immutable partition
Stratified row-based, deterministic random split: 20% final test (seed 42); of remaining 80%, split 62.5% development training versus 37.5% other (seed 314159), divide the latter into separate calibration and selection halves (seed 271828). Global fractions: 50% train, 15% calibration, 15% policy selection, 20% final test. Check exact ID disjointness. **Not a time or customer-grouped test.**

## Frozen candidate models
(A) `LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42)`; (B) `DecisionTreeClassifier(max_depth=6,min_samples_leaf=50,class_weight="balanced",random_state=42)`. Train-only preprocessing: median imputation and StandardScaler for numeric; most-frequent imputation and OneHotEncoder(handle_unknown="ignore") for categorical. Candidate-specific, unweighted sigmoid calibrator is `LogisticRegression(C=1e6,solver="lbfgs",max_iter=2000)` fitted on the **calibration sample alone**, taking model raw-probability logit after clipping to [1e-6,1-1e-6].

## Model and threshold selection
Select model by maximal calibrated-score ROC-AUC on selection sample; tie goes to logistic. Select hypothetical campaign threshold from exactly {0.05, 0.10, ..., 0.95} on this **same selection sample** maximizing `8*true_subscriptions_reached - 1*contacted_customers`; ties favor the higher threshold. These payoff units are arbitrary; all observed customers were contacted, so no incremental, causal, or production value is identified. Do not use final-test labels for selection, calibration or refitting.

## Final untouched report
Report selected model/threshold, split hashes/counts, AUC, Brier, log loss, precision/recall, contacts, observed true responders, illustrative 8*TP-contacts, contact-all 8*all_responders-N, contact-none 0, train-prevalence Brier; exactly 10 equal-width-bin ECE; 1000 final-test bootstrap row resamples (seed 20261013) for percentile 95% CIs of AUC (skip single-class replicate), Brier, and net payoff per 1000. Retain all adverse results. Never retune V2 on final data. Independent matching-source and protocol reproducibility required. No claim of out-of-time transportability, randomized causal uplift, actual profit or calibration beyond the evaluated cohort.

## Abort rules
Abort on malformed dataset/target, missing pre-call leakage exclusion, any split overlap, non-finite model predictions, missing classes, invalid scores, failed reproducibility, and undocumented changes. V1 results and outputs stay byte-for-byte untouched.