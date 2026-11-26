"""Final journal study. Run: python3 final_study.py  (needs ctrl.py, plant2.py)"""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ctrl import *; from plant2 import *
A,B,E,G=mats(**NOM);ev=np.linalg.eigvals(A);t1,t2=sorted(-1/ev,reverse=True);TAU=float((t1+t2/2).real);LL=float((t2/2).real)
MK={'Relay-NTC':lambda:Relay('ntc',46.),'Relay-IR':lambda:Relay('ir',41.),'PI-IR':lambda:PI(41.,18.6,TAU,LL),
    'Observer-LQ':lambda:Obs(41.,qh=.02,qs=1.,rr=3.,qQ=3.,ki=.006)}
def met(Ts,U,bi):
    k0=int(450/DT);k1=int(590/DT);ss=Ts[k0:k1];dq=Ts[int(600/DT):]
    return dict(rise=(np.argmax(Ts>=40.5)*DT) if Ts.max()>=40.5 else np.nan, over=max(0,Ts.max()-41.5), pp=ss.max()-ss.min(),
      rms=np.sqrt(np.mean((ss-41)**2)), rmsb=np.sqrt(np.mean((ss+bi-41)**2)), dip=41-dq.min(), sw=np.sum(np.abs(np.diff(U)))/2, Om=omega(Ts), Tmax=Ts.max())
N=200;rng=np.random.default_rng(11);dr=[]
for i in range(N):
    par={k:v*rng.uniform(.85,1.15) for k,v in NOM.items()}
    d=dict(par=par,Ta=rng.uniform(24,28),bi=rng.normal(0,.25),bn=rng.normal(0,.3),Q=rng.uniform(20,60));d['feas']=d['Ta']+par['Rso']*P-d['Q']*par['Rso']>=41.5;dr.append(d)
fe=[i for i,d in enumerate(dr) if d['feas']]
print('feasible',len(fe),'of',N,'; pre-step headroom<0.5K:',sum(d['Ta']+d['par']['Rso']*P<41.5 for d in dr))
for name,mk in MK.items():
    R=[met(*simulate(mk(),dr[i]['par'],Ta=dr[i]['Ta'],biasIR=dr[i]['bi'],biasNT=dr[i]['bn'],Qstep=dr[i]['Q'],seed=i)[::2],dr[i]['bi']) for i in fe]
    print(name)
    for k in R[0]:
        x=np.array([r[k] for r in R],float);print(f"  {k:5s} {np.nanmean(x):.4g} +- {np.nanstd(x):.3g}  max {np.nanmax(x):.4g}  nan {int(np.isnan(x).sum())}")
# nominal traces + fault traces for the figure
nom=dict(Ta=26.);cols={'Relay-NTC':'#7f7f7f','Relay-IR':'#d95f02','PI-IR':'#1b9e77','Observer-LQ':'#2166ac'}
plt.rcParams.update({"font.family":"serif","font.size":7})
fig,ax=plt.subplots(2,1,figsize=(3.5,3.7),dpi=300)
t=np.arange(int(1200/DT)+1)*DT/60
for n,mk in MK.items():
    Ts,Th,U=simulate(mk(),NOM,Qstep=40.,seed=1); ax[0].plot(t,Ts,color=cols[n],lw=.8,label=n)
ax[0].axhline(41,color='k',lw=.4,ls=':');ax[0].axvline(10,color='k',lw=.4,ls='--');ax[0].text(10.2,37.2,'cold load\n(+40 W)',fontsize=5.5)
ax[0].set_ylim(24,43);ax[0].set_xlim(0,20);ax[0].set_ylabel(r'$T_{\rm surf}$ ($^\circ$C)');ax[0].legend(fontsize=5.5,frameon=False,loc='lower left',ncol=2,bbox_to_anchor=(0.0,0.0));ax[0].set_title('(a) Nominal plant, noise and cold-load step',fontsize=7)
ins=ax[0].inset_axes([.60,.14,.36,.36]);
for n,mk in MK.items():
    Ts,Th,U=simulate(mk(),NOM,Qstep=40.,seed=1);k=(t>7)&(t<9.5);ins.plot(t[k],Ts[k],color=cols[n],lw=.7)
ins.set_ylim(39.5,41.8);ins.tick_params(labelsize=5);ins.set_title('ripple, 7-9.5 min',fontsize=5,pad=1)
for n in ('Relay-IR','Observer-LQ'):
    c=MK[n]();Ts,Th,U=simulate(c,NOM,fault=dict(kind='ir',d=-6.,t=400.),seed=3);ax[1].plot(t,Ts,color=cols[n],lw=.9,label=n)
ax[1].axhline(43.5,color='k',ls='--',lw=.5);ax[1].text(0.4,43.65,'43.5 $^\\circ$C',fontsize=5.5)
ax[1].axvline(400/60,color='r',lw=.4);ax[1].set_ylim(24,46);ax[1].set_xlim(0,20);ax[1].set_xlabel('Time (min)');ax[1].set_ylabel(r'$T_{\rm surf}$ ($^\circ$C)')
ax[1].set_title('(b) IR channel reads 6 K low from t = 6.7 min',fontsize=7);ax[1].legend(fontsize=5.5,frameon=False,loc='upper right',bbox_to_anchor=(1.0,0.78))
fig.tight_layout();fig.savefig('sim_results.png')
c=MK['Observer-LQ']();Ts,Th,U=simulate(c,NOM,fault=dict(kind='ir',d=-6.,t=400.),seed=3);print('obs det',c.det_time,c.kind,'Tmax',Ts.max())
c=MK['Relay-IR']();Ts,Th,U=simulate(c,NOM,fault=dict(kind='ir',d=-6.,t=400.),seed=3);print('relayIR fault Tmax',Ts.max(),'Om',omega(Ts),'t>43.5',(Ts>43.5).sum()*DT)
# detector stats
fa=0;lat=[]
for i in fe[:100]:
    c=MK['Observer-LQ']();simulate(c,dr[i]['par'],Ta=dr[i]['Ta'],biasIR=dr[i]['bi'],biasNT=dr[i]['bn'],Qstep=dr[i]['Q'],seed=i);fa+=c.fault
print('false alarms',fa,'of',len(fe[:100]))
for kind,dd in (('ir',-3.),('ir',-1.5),('ntc',-3.),('ntc',-1.5)):
    L=[];miss=0
    for i in fe[:100]:
        c=MK['Observer-LQ']();simulate(c,dr[i]['par'],Ta=dr[i]['Ta'],biasIR=dr[i]['bi'],biasNT=dr[i]['bn'],fault=dict(kind=kind,d=dd,t=400.),seed=i)
        if c.det_time and c.det_time>=400:L.append(c.det_time-400)
        else: miss+=1
    print(kind,dd,'lat mean',np.mean(L) if L else None,'max',np.max(L) if L else None,'miss',miss,'of',len(fe[:100]))
