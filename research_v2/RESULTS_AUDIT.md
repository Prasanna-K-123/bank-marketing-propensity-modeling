# Registered Bank Policy V2 — independent-holdout evidence and limitations

**Research registration:** `research_v2/REGISTERED_PROTOCOL.md`, originally committed on the study branch at `7ceb41255393da510c2bb76fa48495931151752d` **before** the empirical V2 run. **Source test:** GitHub Actions run `38296515143`, PR #1 initial final-study SHA `a974f4e7cc81659d6d132fa6ba42389a0ae39c60`.

## What was actually observed

UCI Bank Marketing full 45,211 rows, with after-call `duration` excluded (15 predictors). Stratified random, mutually exclusive samples: training 22,605; score calibration 6,781; model/threshold selection 6,782; untouched evaluation 9,043. The source and protocol hashes are recorded in the committed JSON result. Two frozen candidate models; separate sigmoid score calibration; model chosen by selection-set AUC; threshold chosen by selection-set illustrative utility. The held-out test was not used for either decision.

**Selected:** balanced logistic regression, calibrated on its separate sample; hypothetical 0.15 contact-score cutoff. The selection-set logistic AUC was **0.7355**, versus **0.7103** for the tree. On the untouched final test, **ROC-AUC 0.7529** (1,000-row-bootstrap 95% percentile CI **0.7369–0.7697**), Brier **0.08691** (CI **0.08235–0.09134**), log loss **0.30491**, equal-width 10-bin ECE **0.01077**, and fixed train-prevalence Brier **0.10331**. AUC and calibration refer to the **same randomly held-out historical dataset**, not a future-month policy.

At the frozen threshold, **1,390/9,043 (15.37%)** rows selected, of whom **505** were recorded subscribers (precision **36.33%**, recall **47.73%**). Under arbitrary historical net units defined as `8 × selected subscribers − selected contacts`, test score was **+2,650 units**, versus **−579 contact-all** and **zero contact-none** in those hypothetical units. This translates to **293 net units per 1,000 records** (bootstrap interval approximately **261–329**). **These are retrospective accounting proxies, not realized or incremental economic gains.** The source contains outcomes after the historical calls; there is no uncontacted counterfactual, random treatment, operational targeting experiment, true offer cost, or observed conversion uplift. These numbers cannot support a revenue/P&L/ROI claim.

## What remains genuinely unresolved

- Row-stratified random splitting is not an out-of-time holdout, may put related records in different partitions, and gives no prospective deployment/transport guarantee.
- The only fixed model candidates were a balanced logistic classifier and depth-constrained tree. Their predictive ranking/cost tradeoffs can change with a different sample, access-to-feature timing, target definition, or business price/cost structure.
- A bootstrap interval for final-test predictive metrics captures sampling variation conditional on the frozen fitted candidate and selected threshold; it does not capture model-selection uncertainty, treatment effect, covariate shift or repeated use of the test.
- The measured ECE is a bin-dependent diagnostic, not proof of calibration for customer subgroups or future campaigns.
- The original V1 exploratory ROC-AUC **0.755** and threshold **0.55** remain a different split/evaluation protocol. They must not be described as externally replicated, superseded by a comparable V2 controlled experiment, or used to claim performance improvement.
- Results come from UCI historical sample; the economic-policy evaluation requires actual costs, counterfactual noncontacts and eventually a randomized controlled holdout before causal value claims.

## Reproduction and evidence boundaries

Run `python -m pip install -r research_v2/requirements.txt`, `python -m pytest research_v2/test_registered_policy.py -q`, then `python research_v2/validate_policy.py`. The command refuses to overwrite an existing, different `research_v2/results/registered_policy_v2.json`, and CI validates the committed full report. Source inputs are fetched live from UCI id 222 and checked against a fixed dataset fingerprint. This is an additional registered research evaluation, *not* a replacement or retroactive correction of V1. Frozen historical V1 script/results, source identity and their limitations are preserved. No HR/interview competence, external review or deployment has been assessed.
