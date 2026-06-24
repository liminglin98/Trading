"""Calibrate the condor10 sleeve: edge-bearing geometry (shorts ~1 sigma OTM),
find the fixed bankroll fraction per trade whose 2015-2026 backtest CAGR ~= 10%,
under both flat-vol and real-skew pricing. Writes out/condor10_spec.json."""
import pandas as pd, numpy as np, json
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

DATA="/sessions/eloquent-fervent-goodall/mnt/Trading/Data/akshare/"
QM="/sessions/eloquent-fervent-goodall/mnt/Trading/qmt_export/"
OUT="/sessions/eloquent-fervent-goodall/mnt/Trading/analysis/macro_model/out/"
R=0.017
def d1(S,K,s,T): return (log(S/K)+(R+0.5*s*s)*T)/(s*sqrt(T))
def cpx(S,K,s,T):
    if T<=0 or s<=0: return max(S-K,0)
    a=d1(S,K,s,T); return S*norm.cdf(a)-K*exp(-R*T)*norm.cdf(a-s*sqrt(T))
def ppx(S,K,s,T):
    if T<=0 or s<=0: return max(K-S,0)
    a=d1(S,K,s,T); return K*exp(-R*T)*norm.cdf(-(a-s*sqrt(T)))-S*norm.cdf(-a)
def ivp(px,S,K,T,typ):
    try: return brentq(lambda s:(cpx if typ=='C' else ppx)(S,K,s,T)-px,1e-3,5,xtol=1e-5)
    except Exception: return np.nan

# --- real ~30d skew from the July chain ---
m=pd.read_csv(QM+"option_metadata.csv",dtype={'contract':str})
d=pd.read_csv(QM+"option_daily.csv",dtype={'contract':str})
day=d[d.date==20260622][['contract','close']]
mm=m.merge(day,on='contract'); mm=mm[mm.ExpireDate==20260722].copy()
S0=3.097; T0=30/365.0; mm['typ']=np.where(mm.optType=='CALL','C','P')
mm['iv']=[ivp(r.close,S0,r.OptExercisePrice,T0,r.typ) for r in mm.itertuples()]
mm['x']=np.log(mm.OptExercisePrice/S0)
otm=mm[((mm.typ=='P')&(mm.x<0))|((mm.typ=='C')&(mm.x>=0))].dropna(subset=['iv']).sort_values('x')
atm=np.interp(0,otm.x,otm.iv); coef=np.polyfit(otm.x,otm.iv/atm,2)
skew=lambda x: float(np.clip(np.polyval(coef,x),0.5,2.5))

# --- history ---
def load(f,c):
    z=pd.read_csv(DATA+f); z.columns=[k.strip().lstrip('﻿') for k in z.columns]
    z['date']=pd.to_datetime(z['date']); return z[['date',c]].dropna()
u=load("etf_510050_daily.csv","close"); q=load("qvix_50etf.csv","close").rename(columns={'close':'qvix'})
df=u.merge(q,on='date').sort_values('date').set_index('date'); idx=df.index
nob=lambda ts:(idx[idx.searchsorted(ts,'right')-1] if idx.searchsorted(ts,'right')-1>=0 else None)
def fw(y,mo):
    dd=pd.Timestamp(y,mo,1); return dd+pd.Timedelta(days=(2-dd.weekday())%7)+pd.Timedelta(weeks=3)
exps=[fw(y,mo) for y in range(2015,2027) for mo in range(1,13)]
exps=[e for e in exps if df.index.min()<=e<=df.index.max()]
rstk=lambda K:round(K/0.05)*0.05
Kc_d=lambda S,dl,s,T:S/exp(norm.ppf(dl)*s*sqrt(T)-(R+0.5*s*s)*T)
Kp_d=lambda S,dl,s,T:S/exp(norm.ppf(1-dl)*s*sqrt(T)-(R+0.5*s*s)*T)

def backtest(delta=0.20, wing=0.15, use_skew=True):
    rr=[]
    for e in exps:
        ed=nob(e); t0=nob(e-pd.Timedelta(days=30))
        if ed is None or t0 is None or t0>=ed: continue
        S=df.loc[t0,'close']; sig=df.loc[t0,'qvix']/100; T=(ed-t0).days/365
        if T<=0 or sig<=0: continue
        Kc=rstk(Kc_d(S,delta,sig,T)); Kp=rstk(Kp_d(S,delta,sig,T))
        Kcl=rstk(Kc+wing); Kpl=rstk(Kp-wing)
        if Kp>=Kc: continue
        def P(K,typ,Sx=S,Tx=T,sg=sig):
            s=sg*(skew(log(K/Sx)) if use_skew else 1.0); return (cpx if typ=='C' else ppx)(Sx,K,s,Tx)
        credit=P(Kc,'C')+P(Kp,'P')-P(Kcl,'C')-P(Kpl,'P'); risk=max(wing-credit,1e-6)
        ST=df.loc[ed,'close']
        pay=(max(ST-Kc,0)-max(ST-Kcl,0))+(max(Kp-ST,0)-max(Kpl-ST,0))
        rr.append((credit-pay)/risk)
    return np.array(rr)

def cagr(r,f):
    g=1+f*r
    if (g<=0).any(): return -1.0,None
    eq=np.cumprod(g); return eq[-1]**(12/len(r))-1, eq
def mdd(eq): return float((eq/np.maximum.accumulate(eq)-1).min())

def calibrate(r,target=0.10):
    fs=np.arange(0.01,0.96,0.005); cg=np.array([cagr(r,f)[0] for f in fs])
    hit=np.where(cg>=target)[0]
    if len(hit)==0: return None
    f=float(fs[hit[0]]); _,eq=cagr(r,f)
    return dict(fraction=round(f,3), cagr=round(cagr(r,f)[0],4), maxDD=round(mdd(eq),3),
               worst_month=round(float(f*r.min()),3), win=round(float((r>0).mean()),3),
               mean_r_on_risk=round(float(r.mean()),4), n=len(r))

spec={"geometry":{"short_steps_OTM":2,"protective_steps":3,"approx_short_delta":0.20,
                  "approx_short_OTM_pct":5.0,"wing_price_width":0.15},
      "target_annual_return":0.10,
      "calibration_sample":"50ETF 2015-02..2026-06, 135 monthly expiries, BS off QVIX",
      "flat_vol":calibrate(backtest(use_skew=False)),
      "real_skew":calibrate(backtest(use_skew=True)),
      "notes":"Sizing fraction = bankroll fraction risked per trade (max_loss based). "
              "Edge is near-the-money/VRP, thin and regime-dependent; far-OTM selling had "
              "~zero/negative edge in-sample. Treat 10% as a SIZING target on an uncertain "
              "edge, not a guaranteed yield. Real skew thins credit vs flat-vol."}
print(json.dumps(spec,indent=2))
json.dump(spec,open(OUT+"condor10_spec.json","w"),indent=2)
print("\nwrote",OUT+"condor10_spec.json")
