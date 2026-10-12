"""Registered V2 retrospective bank-response policy validation. Never modifies V1 outputs.

The protocol is research_v2/REGISTERED_PROTOCOL.md, committed before a V2 test run.
Do not use this script to claim a causal effect of targeting.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / 'research_v2' / 'results'
THRESHOLDS = tuple(round(i / 100, 2) for i in range(5, 96, 5))


def validate_source(raw: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(raw) != 45211 or len(target) != 45211:
        raise ValueError('Dataset must contain exactly 45,211 full Bank Marketing rows')
    if list(raw.columns).count('duration') != 1:
        raise ValueError('Expected exactly one call-duration feature')
    y = target.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    if y.isna().any() or y.nunique() != 2:
        raise ValueError('Unexpected binary target; abort')
    if raw.columns.duplicated().any():
        raise ValueError('Duplicate feature columns')
    X = raw.drop(columns=['duration']).copy()
    if X.shape[1] != 15:
        raise ValueError('Expected 15 pre-call features after excluding duration')
    return X.reset_index(drop=True), y.reset_index(drop=True).astype(int)


def partition_indices(y: pd.Series) -> dict[str, np.ndarray]:
    idx = np.arange(len(y))
    dev, test = train_test_split(idx, test_size=.20, random_state=42, stratify=y)
    train, other = train_test_split(dev, train_size=.625, random_state=314159,
                                   stratify=y.iloc[dev])
    calibration, selection = train_test_split(other, test_size=.50,
                                              random_state=271828,
                                              stratify=y.iloc[other])
    blocks = dict(train=np.sort(train), calibration=np.sort(calibration),
                  selection=np.sort(selection), test=np.sort(test))
    if len(np.unique(np.concatenate(list(blocks.values())))) != len(y):
        raise AssertionError('Partitions overlap or omit records')
    if any(y.iloc[v].nunique() != 2 for v in blocks.values()):
        raise ValueError('Insufficient class variation in a partition')
    return blocks


def make_model(X_train: pd.DataFrame, key: str) -> Pipeline:
    numerical = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numerical]
    pre = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numerical),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    if key == 'logistic':
        estimator = LogisticRegression(max_iter=2000, class_weight='balanced',
                                       random_state=42)
    elif key == 'tree':
        estimator = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                            class_weight='balanced', random_state=42)
    else:
        raise ValueError('Unregistered candidate')
    return Pipeline([('preprocess', pre), ('estimator', estimator)])


def logits(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    score = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError('Model returned invalid raw scores')
    p = np.clip(score, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrated(model: Pipeline, calibrator: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    p = np.asarray(calibrator.predict_proba(logits(model, X))[:, 1], dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid calibrated probabilities')
    return p


def policy_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    selected = p >= threshold
    tp = int((selected & (y == 1)).sum())
    n = int(len(y))
    contacts = int(selected.sum())
    return dict(threshold=threshold, contacts=contacts, contact_fraction=contacts / n,
                true_subscribers_reached=tp, precision=float(precision_score(y, selected, zero_division=0)),
                recall=float(recall_score(y, selected, zero_division=0)),
                illustrative_net_units=8 * tp - contacts,
                contact_all_net_units=8 * int(y.sum()) - n,
                contact_none_net_units=0)


def choose_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, list[dict]]:
    rows = [policy_metrics(y, p, t) for t in THRESHOLDS]
    # Deterministic tie handling: larger threshold wins.
    choice = max(rows, key=lambda row: (row['illustrative_net_units'], row['threshold']))
    return float(choice['threshold']), rows


def ece(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.digitize(p, np.linspace(0, 1, 11)[1:-1], right=False)
    out = 0.0
    for b in range(10):
        ix = bins == b
        if ix.any():
            out += float(ix.mean()) * abs(float(y[ix].mean()) - float(p[ix].mean()))
    return float(out)


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float, n_rep: int = 1000) -> dict:
    rng = np.random.default_rng(20261013)
    measures = {'roc_auc': [], 'brier': [], 'net_units_per_1000': []}
    for _ in range(n_rep):
        ids = rng.integers(0, len(y), len(y))
        yy, pp = y[ids], p[ids]
        if len(np.unique(yy)) == 2:
            measures['roc_auc'].append(float(roc_auc_score(yy, pp)))
        measures['brier'].append(float(brier_score_loss(yy, pp)))
        measures['net_units_per_1000'].append(
            float(policy_metrics(yy, pp, threshold)['illustrative_net_units']) / len(yy) * 1000)
    return {name: dict(low=float(np.percentile(values, 2.5)),
                       high=float(np.percentile(values, 97.5)),
                       replicates=len(values)) for name, values in measures.items()}


def evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    blocks = partition_indices(y)
    model_summary = {}
    trained = {}
    for key in ('logistic', 'tree'):
        model = make_model(X.iloc[blocks['train']], key)
        model.fit(X.iloc[blocks['train']], y.iloc[blocks['train']])
        calibrator = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000)
        calibrator.fit(logits(model, X.iloc[blocks['calibration']]), y.iloc[blocks['calibration']])
        selection_score = calibrated(model, calibrator, X.iloc[blocks['selection']])
        auc = float(roc_auc_score(y.iloc[blocks['selection']], selection_score))
        model_summary[key] = {'selection_roc_auc': auc}
        trained[key] = (model, calibrator, selection_score)
    selected = max(('logistic', 'tree'), key=lambda name: (
        model_summary[name]['selection_roc_auc'], name == 'logistic'))
    model, calibrator, selection_score = trained[selected]
    selection_y = y.iloc[blocks['selection']].to_numpy()
    threshold, selection_table = choose_threshold(selection_y, selection_score)
    # FINAL TEST LABELS FIRST REFERENCED AFTER CHOICE HAS BEEN FROZEN.
    final_y = y.iloc[blocks['test']].to_numpy()
    final_p = calibrated(model, calibrator, X.iloc[blocks['test']])
    final = policy_metrics(final_y, final_p, threshold)
    final.update(roc_auc=float(roc_auc_score(final_y, final_p)),
                 brier=float(brier_score_loss(final_y, final_p)),
                 log_loss=float(log_loss(final_y, final_p, labels=[0, 1])),
                 ece_10_bins=ece(final_y, final_p),
                 train_prevalence_brier=float(brier_score_loss(final_y,
                     np.full(len(final_y), float(y.iloc[blocks['train']].mean())))),
                 bootstrap_1000=bootstrap_ci(final_y, final_p, threshold))
    return {'selection': {'selected_model': selected, 'selected_threshold': threshold,
                          'model_selection': model_summary, 'threshold_table': selection_table},
            'final_untouched_test': final,
            'partitions': {name: {'n': len(indices), 'indices_sha256':
                           hashlib.sha256(indices.tobytes()).hexdigest(),
                           'prevalence': float(y.iloc[indices].mean())} for name, indices in blocks.items()}}


def main():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=222)
    raw = ds.data.features.copy()
    target = ds.data.targets.iloc[:, 0].copy()
    X, y = validate_source(raw, target)
    source_data = pd.concat([raw.reset_index(drop=True), target.rename('TARGET').reset_index(drop=True)], axis=1)
    source_hash = hashlib.sha256(source_data.to_csv(index=False).encode('utf8')).hexdigest()
    protocol_path = PROJECT_ROOT / 'research_v2' / 'REGISTERED_PROTOCOL.md'
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    result = evaluate(X, y)
    result['provenance'] = dict(dataset_id=222, dataset_sha256=source_hash,
        protocol_sha256=protocol_hash, row_count=len(X), predictor_count=X.shape[1],
        python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
        pandas=pd.__version__, note='Retrospective random-split association; not causal uplift or actual profit.')
    OUT.mkdir(parents=True, exist_ok=True)
    file = OUT / 'registered_policy_v2.json'
    if file.exists():
        old = json.loads(file.read_text())
        if old != result:
            raise RuntimeError('Frozen V2 result changed: refuse to overwrite')
    else:
        file.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n', encoding='utf8')
    print('REGISTERED_V2_FINAL='+json.dumps({
        'model':result['selection']['selected_model'],
        'threshold':result['selection']['selected_threshold'],
        'roc_auc':result['final_untouched_test']['roc_auc'],
        'brier':result['final_untouched_test']['brier'],
        'net_units':result['final_untouched_test']['illustrative_net_units'],
        'contact_all_net_units':result['final_untouched_test']['contact_all_net_units'],
        'data_sha256':source_hash,
        'protocol_sha256':protocol_hash,
    },sort_keys=True))


if __name__ == '__main__':
    main()