"""Neural vessel trajectory forecasters, trained from scratch."""
import torch
from torch import nn

class Forecaster(nn.Module):
    def __init__(self,kind='gru',hidden=32,horizon=12):
        super().__init__()
        if kind not in ('gru','lstm','hybrid_gru'): raise ValueError(kind)
        self.kind=kind
        recurrent=nn.GRU if kind!='lstm' else nn.LSTM
        self.rnn=recurrent(input_size=2,hidden_size=hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden,64),nn.ReLU(),nn.Linear(64,horizon*2))
        self.horizon=horizon
    def forward(self,x,baseline=None):
        out,_=self.rnn(x)
        pred=self.head(out[:,-1]).reshape(-1,self.horizon,2)
        if self.kind=='hybrid_gru':
            if baseline is None: raise ValueError('hybrid requires baseline')
            return baseline+pred
        return pred
