"""Train and evaluate on disjoint vessel seeds; never train on test trajectories."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import make_dataset
from .models import Forecaster

def errors(pred,truth):
    d=np.linalg.norm((pred-truth)*100,axis=-1)
    return {'ADE_m':round(float(d.mean()),2),'FDE_m':round(float(d[:,-1].mean()),2),'RMSE_m':round(float(np.sqrt(np.mean((pred-truth)**2)*2)*100),2)}

def train(model,train_data,val_data,epochs=12,batch=128):
    X,Y,B,*_=train_data
    torch.set_num_threads(2)
    dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(Y),torch.from_numpy(B)),batch_size=batch,shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=0.002)
    loss_fn=nn.MSELoss()
    best=float('inf');state=None;wait=0;history=[]
    for epoch in range(epochs):
        model.train(); total=0
        for x,y,b in dl:
            opt.zero_grad();pred=model(x,b);loss=loss_fn(pred,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();total+=loss.item()*len(x)
        model.eval()
        with torch.no_grad():
            xv,yv,bv,*_=val_data
            vl=loss_fn(model(torch.from_numpy(xv),torch.from_numpy(bv)),torch.from_numpy(yv)).item()
        history.append((total/len(X),vl))
        if vl<best-1e-6:
            best=vl;state={k:v.detach().clone() for k,v in model.state_dict().items()};wait=0
        else:
            wait+=1
            if wait>=5: break
    model.load_state_dict(state)
    return np.array(history)

def predict(model,data):
    x,_,b,*_=data
    model.eval()
    with torch.no_grad(): return model(torch.from_numpy(x),torch.from_numpy(b)).numpy()

def run(output='results',epochs=12,seed=42,train_vessels=65):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Vessel-level split; different seeds/trajectories across splits.
    tr=make_dataset(range(100,100+train_vessels))
    va=make_dataset(range(1000,1015))
    te=make_dataset(range(2000,2020))
    X,Y,B,truth,origins=te
    metrics={'constant_velocity':errors(B,Y)}
    curves={};preds={'constant_velocity':B}
    for kind in ('gru','lstm','hybrid_gru'):
        model=Forecaster(kind)
        curves[kind]=train(model,tr,va,epochs=epochs)
        preds[kind]=predict(model,te)
        metrics[kind]=errors(preds[kind],Y)
        torch.save(model.state_dict(),output/f'{kind}.pt')
    (output/'metrics.json').write_text(json.dumps({'data':'synthetic only','seed':seed,'train_vessels':train_vessels,'validation_vessels':15,'test_vessels':20,'forecast_seconds':60,'metrics':metrics},indent=2))
    # Compare trajectory forecast on one unseen vessel window.
    idx=10
    plt.figure(figsize=(9,6))
    plt.plot(truth[idx,:,0],truth[idx,:,1],'-o',label='Ground truth',lw=2)
    for name,p in preds.items():
        xy=origins[idx]+100*p[idx]
        plt.plot(xy[:,0],xy[:,1],label=name.replace('_',' ').title(),alpha=.8)
    plt.scatter(*origins[idx],marker='*',s=110,label='Last observation')
    plt.xlabel('East (m)');plt.ylabel('North (m)');plt.title('Unseen synthetic vessel: 60-second forecasts');plt.legend(fontsize=8);plt.axis('equal');plt.grid(alpha=.25);plt.tight_layout();plt.savefig(output/'01_forecast_comparison.png',dpi=160);plt.close()
    names=list(metrics)
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar(names,[metrics[k]['ADE_m'] for k in names]);ax.set_ylabel('ADE (m), lower is better');ax.set_title('Test-set forecast accuracy — disjoint vessel tracks')
    ax.tick_params(axis='x',rotation=15)
    for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{bar.get_height():.1f}',ha='center',va='bottom')
    fig.tight_layout();fig.savefig(output/'02_ade_benchmark.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for name,c in curves.items():ax.plot(c[:,1],label=name.replace('_',' ').title())
    ax.set(xlabel='Epoch (zero indexed)',ylabel='Validation MSE (scaled positions)',title='Learning curves — validation vessels');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'03_learning_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for name,p in preds.items():ax.plot(np.arange(1,13)*5,np.linalg.norm((p-Y)*100,axis=-1).mean(axis=0),label=name.replace('_',' ').title())
    ax.set(xlabel='Forecast horizon (s)',ylabel='Mean displacement error (m)',title='Forecast error vs horizon');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(output/'04_horizon_error.png',dpi=160);plt.close(fig)
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=12);p.add_argument('--output',default='results');p.add_argument('--seed',type=int,default=42);p.add_argument('--train-vessels',type=int,default=65)
    args=p.parse_args();run(args.output,args.epochs,args.seed,args.train_vessels)
