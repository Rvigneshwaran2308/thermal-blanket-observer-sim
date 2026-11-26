import numpy as np
from scipy.linalg import solve_discrete_are
from plant2 import *
A_,A2=0,0
class PWM:
    def __init__(s,win=16): s.win=win; s.k=0; s.on=0; s.carry=0.
    def step(s,duty):
        if s.k==0:
            x=duty*s.win+s.carry; s.on=int(np.clip(round(x),0,s.win)); s.carry=x-s.on
        out=1 if s.k<s.on else 0; s.k=(s.k+1)%s.win; return out
class Relay:
    def __init__(s,key,set_,db=.5): s.key=key;s.set=set_;s.db=db;s.u=0
    def __call__(s,m,k):
        y=m[s.key]
        if y<=s.set-s.db: s.u=1
        elif y>=s.set+s.db: s.u=0
        return s.u
class PI:
    def __init__(s,r,K,tau,L,tc=None):
        tc=tc if tc else max(L,8.)
        s.Kc=tau/(K*(tc+L)); s.Ti=min(tau,4*(tc+L)); s.r=r; s.I=0.; s.pwm=PWM(); s.K=K
    def __call__(s,m,k):
        e=s.r-m['ir']; v=s.Kc*e+s.I; u=np.clip(v,0,1)
        s.I+= DT*(s.Kc/s.Ti*e + (u-v)*0.5/ s.Ti*s.Kc*0+ (u-v)/max(s.Ti,1.)*1.0)   # back-calc anti-windup
        return s.pwm.step(u)
class Obs:
    """Kalman observer on [Th,Ts,Q] (Q = unmeasured surface heat sink) + LQ state feedback + plane governor + residual FDI."""
    def __init__(s,r,qh=0.02,qs=1.0,rr=40.,qQ=0.4,Thmax=47.5,nom=NOM,sig_ir=.12,sig_nt=.08,ki=0.0):
        Ad,Bd,Ed,Gd=disc(**nom); s.Ad,s.Bd,s.Ed,s.Gd=Ad,Bd,Ed,Gd
        F=np.zeros((3,3)); F[:2,:2]=Ad; F[:2,2]=Gd; F[2,2]=1
        H=np.array([[1,0,0],[0,1,0]],float)
        Qn=np.diag([1e-4,1e-4,qQ*DT])
        Rn=np.diag([sig_nt**2,sig_ir**2])
        Pm=solve_discrete_are(F.T,H.T,Qn,Rn); s.F=F;s.H=H
        s.Sinn=H@Pm@H.T+Rn; s.Kf=Pm@H.T@np.linalg.inv(s.Sinn)
        Pc=solve_discrete_are(Ad,Bd.reshape(2,1),np.diag([qh,qs]),np.array([[rr]]))
        s.Kx=np.linalg.solve(Bd.reshape(1,2)@Pc@Bd.reshape(2,1)+rr, Bd.reshape(1,2)@Pc@Ad)[0]
        s.ki=ki;s.I=0.;s.r=r;s.x=np.array([26.,26.,0.]);s.uprev=0.;s.pwm=PWM();s.Thmax=Thmax;s.nom=nom
        s.nu=np.zeros(2);s.fault=False;s.cs=0;s.det_time=None;s.cnt=0;s.hist=[];s.gcnt=0;s.kind=None;s.rg=0.
    def __call__(s,m,k):
        Ta=m['ta']
        # predict
        xp=s.F@s.x; xp[:2]+=s.Bd*s.uprev+s.Ed*Ta
        z=np.array([m['ntc'],m['ir']]); nu=z-s.H@xp; s.x=xp+s.Kf@nu; s.nu=nu
        # residual FDI: normalized innovation, two-sided, consecutive-sample rule
        sd=np.sqrt(np.diag(s.Sinn)); zn=np.abs(nu)/sd
        s.cnt = s.cnt+1 if ((zn>6.0)&(np.abs(nu)>1.0)).any() else 0
        if s.cnt>=12 and not s.fault: s.fault=True; s.det_time=k*DT; s.kind='innov'
        s.hist.append((m['ntc'],m['ir'],s.uprev))
        if len(s.hist)>240: s.hist.pop(0)
        if k%16==0 and len(s.hist)==240 and not s.fault:
            h=np.array(s.hist); tt=np.arange(240)*DT
            slope=np.polyfit(tt,h[:,0],1)[0]
            g=np.mean(h[:,0]-h[:,1]); s.rg=g-s.nom['Rhs']*(P*np.mean(h[:,2])-2600.*slope)
            s.gcnt = s.gcnt+1 if abs(s.rg)>2.0 else 0
            if s.gcnt>=3: s.fault=True; s.det_time=k*DT; s.kind='energy'
        if s.fault: s.uprev=0.; return 0
        Rso=s.nom['Rso'];Rhs=s.nom['Rhs']
        Qh=s.x[2]; uss=np.clip(((s.r-Ta)/Rso+Qh)/P,0,1); Thref=s.r+Rhs*P*uss
        e_s=s.r-s.x[1]
        u_raw=uss+s.I-s.Kx@(s.x[:2]-np.array([Thref,s.r]))
        u=float(np.clip(u_raw,0,1))
        if abs(e_s)<2.0 and u_raw==u: s.I=float(np.clip(s.I+DT*s.ki*e_s,-0.3,0.3))
        if s.x[0]>s.Thmax: u=0.
        a=s.pwm.step(u); s.uprev=float(a); return a
def simulate(ctrl,par,Ta=26.,biasIR=0.,biasNT=0.,sIR=.12,sNT=.08,sTa=.3,T=1200.,Qstep=0.,tQ=600.,fault=None,seed=0):
    rng=np.random.default_rng(seed)
    Ad,Bd,Ed,Gd=disc(**par); n=int(T/DT); x=np.array([Ta,Ta]); Ts=np.zeros(n+1);Th=np.zeros(n+1);U=np.zeros(n);Ts[0]=Th[0]=Ta
    for k in range(n):
        ir=x[1]+biasIR+rng.normal(0,sIR); nt=x[0]+biasNT+rng.normal(0,sNT)
        if fault and k*DT>=fault['t']:
            if fault['kind']=='ir': ir+=fault['d']
            if fault['kind']=='ntc': nt+=fault['d']
        m=dict(ir=ir,ntc=nt,ta=Ta+rng.normal(0,sTa))
        # cutoffs (paper): measured surf>=43.5 or sub>=50 -> latch off
        if not hasattr(simulate,'_l') or k==0: simulate._l=False
        if ir>=43.5 or nt>=50.0: simulate._l=True
        u=0 if simulate._l else ctrl(m,k)
        U[k]=u; Q=Qstep if k*DT>=tQ else 0.
        x=Ad@x+Bd*u+Ed*Ta+Gd*Q; Ts[k+1]=x[1];Th[k+1]=x[0]
    return Ts,Th,U
A_=3.1e98;dE=6.28e5;Rg=8.314
def omega(Ts): return float(np.sum(A_*np.exp(-dE/(Rg*(Ts+273.15)))*DT))
