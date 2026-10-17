"""Generate disjoint synthetic vessel tracks and supervised windows."""
import numpy as np

def vessel(seed, n=105, dt=5.0, noise=8.0):
    rng=np.random.default_rng(seed)
    t=np.arange(n)*dt
    speed0=rng.uniform(3.5,10)
    heading=rng.uniform(-np.pi,np.pi)
    mode=seed%5
    xy=np.zeros((n,2),dtype=np.float32)
    vel=np.zeros((n,2),dtype=np.float32)
    for k in range(n):
        turn=(0.0 if mode==0 else (0.0018*np.sin(k/13+seed) if mode==1 else 0.0035*np.sin(k/18+seed) if mode==2 else 0.004*(1 if k>40 else 0) if mode==3 else 0.006*np.sin(k/9+seed)))
        heading+=turn*dt
        speed=np.clip(speed0+0.8*np.sin(k/17+seed)+ (0.7*np.sin(k/9) if mode==4 else 0),2,13)
        vel[k]=[speed*np.sin(heading),speed*np.cos(heading)]
        if k: xy[k]=xy[k-1]+vel[k-1]*dt
    obs=xy+rng.normal(0,noise,xy.shape).astype(np.float32)
    return xy,obs

def make_dataset(seeds, history=12, horizon=12, dt=5.0):
    X=[];Y=[];B=[];truth=[];origins=[]
    for seed in seeds:
        xy,obs=vessel(int(seed),dt=dt)
        for k in range(history-1,len(xy)-horizon,3):
            hist=obs[k-history+1:k+1]
            # Velocity estimate from end-to-end displacement is robust to position noise.
            v=(hist[-1]-hist[0])/((history-1)*dt)
            baseline=np.arange(1,horizon+1,dtype=np.float32)[:,None]*dt*v
            # Relative observed history in metres, scaled for neural optimization.
            X.append((hist-hist[-1])/100.0)
            Y.append((xy[k+1:k+1+horizon]-obs[k])/100.0)
            B.append(baseline/100.0)
            truth.append(xy[k+1:k+1+horizon])
            origins.append(obs[k])
    return tuple(np.asarray(a,dtype=np.float32) for a in (X,Y,B,truth,origins))
