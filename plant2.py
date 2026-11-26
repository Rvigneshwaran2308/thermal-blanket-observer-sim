import numpy as np
from scipy.linalg import expm, solve_discrete_are
P=550.; DT=0.25
NOM=dict(Rso=18.6/P, Rhs=0.0113, Ch=2600., Cs=1500.)
def mats(Rso,Rhs,Ch,Cs):
    A=np.array([[-1/(Rhs*Ch), 1/(Rhs*Ch)],[1/(Rhs*Cs), -(1/Rhs+1/Rso)/Cs]])
    B=np.array([P/Ch,0.]); E=np.array([0.,1/(Rso*Cs)]); G=np.array([0.,-1/Cs])   # Ta input, Q sink input
    return A,B,E,G
def disc(Rso,Rhs,Ch,Cs,dt=DT):
    A,B,E,G=mats(Rso,Rhs,Ch,Cs)
    M=np.zeros((5,5)); M[:2,:2]=A; M[:2,2]=B; M[:2,3]=E; M[:2,4]=G
    Md=expm(M*dt); return Md[:2,:2],Md[:2,2],Md[:2,3],Md[:2,4]
if __name__=="__main__":
    A,B,E,G=mats(**NOM); ev=np.linalg.eigvals(A); print('poles',ev,'taus',-1/ev)
    t1,t2=sorted(-1/ev,reverse=True)
    print('half-rule tau',t1+t2/2,'L',t2/2)
    # DC gains
    Rso,Rhs=NOM['Rso'],NOM['Rhs']
    print('Ks',Rso*P,'Kh',(Rso+Rhs)*P)
    # step response sim: time to 15K rise, plane temp then
    Ad,Bd,Ed,Gd=disc(**NOM);x=np.array([26.,26.]);t=0
    while x[1]<41: x=Ad@x+Bd*1+Ed*26;t+=DT
    print('rise',t,'Th at rise',x[0])
