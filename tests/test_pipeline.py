import numpy as np
import torch
from maritime_hybrid.data import make_dataset,vessel
from maritime_hybrid.models import Forecaster
from maritime_hybrid.experiment import errors

def test_trajectories():
    a,b=vessel(12)
    assert a.shape==b.shape==(105,2)
    assert np.isfinite(b).all()

def test_dataset():
    x,y,b,*_=make_dataset(range(2))
    assert x.shape[1:]==(12,2) and y.shape[1:]==(12,2)
    assert len(x)==len(b)

def test_models():
    for name in ('gru','lstm','hybrid_gru'):
        m=Forecaster(name)
        p=m(torch.zeros(3,12,2),torch.zeros(3,12,2))
        assert p.shape==(3,12,2)

def test_metrics():
    z=np.zeros((2,12,2))
    assert errors(z,z)['ADE_m']==0
