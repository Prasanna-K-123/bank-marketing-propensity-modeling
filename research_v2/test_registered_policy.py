import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pytest
from validate_policy import (THRESHOLDS, bootstrap_ci, choose_threshold,
                             ece, evaluate, partition_indices, policy_metrics,
                             validate_source)


def test_source_contract_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match='45,211'):
        validate_source(pd.DataFrame({'duration':[1,2]}), pd.Series(['yes','no']))


def test_full_split_disjoint_reproducible_and_stratified():
    y=pd.Series(np.tile([0,1,0,1,0,1,0,0,0,1],500))
    parts=partition_indices(y)
    parts2=partition_indices(y)
    assert sum(map(len, parts.values())) == len(y)
    assert len(set(np.concatenate(list(parts.values()))))==len(y)
    assert all(np.array_equal(parts[k],parts2[k]) for k in parts)
    assert len(parts['test'])==1000
    assert len(parts['train'])==2500
    assert all(abs(float(y.iloc[ids].mean())-float(y.mean())) < .02 for ids in parts.values())


def test_policy_arithmetic_and_threshold_ties():
    y=np.array([1,0,1,0]);p=np.array([.9,.7,.3,.1])
    row=policy_metrics(y,p,.5)
    assert row['contacts']==2 and row['true_subscribers_reached']==1
    assert row['illustrative_net_units']==6 and row['contact_all_net_units']==12
    assert len(THRESHOLDS)==19
    assert all(0 < x < 1 for x in THRESHOLDS)
    y=np.array([0,0]); p=np.array([.1,.2])
    # All thresholds >=0.25 tie at zero contacts and zero payoff.
    choice, _=choose_threshold(y,p)
    assert choice==.95


def test_binned_calibration_exact():
    y=np.array([0,1]);p=np.array([.1,.9])
    assert ece(y,p)==pytest.approx(.1)
    assert ece(np.array([0,1]),np.array([0.,1.]))==0.


def test_bootstrap_deterministic_bounds():
    y=np.array([0,0,1,1,0,1,0,1]*10)
    p=np.array([.15,.35,.75,.85,.4,.7,.2,.8]*10)
    a=bootstrap_ci(y,p,.5,n_rep=30)
    b=bootstrap_ci(y,p,.5,n_rep=30)
    assert a==b
    assert all(v['low']<=v['high'] for v in a.values())
    assert a['roc_auc']['replicates']==30


def test_end_to_end_synthetic_no_final_selection_and_calibration():
    rng=np.random.default_rng(1234);n=400
    feature=rng.standard_normal(n)
    y=pd.Series((rng.random(n)<1/(1+np.exp(-feature))).astype(int))
    X=pd.DataFrame({'number':feature,'category':np.where(feature>0,'one','other')})
    result=evaluate(X,y)
    assert result['selection']['selected_model'] in ('tree','logistic')
    assert result['selection']['selected_threshold'] in THRESHOLDS
    assert len(result['selection']['threshold_table'])==19
    assert result['final_untouched_test']['contacts']<=80
    assert result['final_untouched_test']['bootstrap_1000']['brier']['replicates']==1000
    assert all(0<=result['final_untouched_test'][name]<=1 for name in ('roc_auc','brier','ece_10_bins'))
