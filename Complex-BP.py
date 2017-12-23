#Implementation of BackPropogation Algorithm Using Complex Number for geometric transformation
#1-6-1 3 layered network is used

import numpy as np
import matplotlib.pyplot as plt
class Neural:
    def __init__(self, Inp):  
        self.ni=1
        self.nh=6
        self.no=1
        #initializing complex weight/threshold from input layer to hidden layer
        self.Wji=np.zeros(6,dtype=complex)
        for l in range(self.nh):
           self.Wji[l]=np.random.random()+np.random.random()*1j
        
        #  self.changenoenoe=10 #no of iterations
        self.Tj=np.zeros(6,dtype=complex)
        
        ran=np.random.random()+np.random.random()*1j
        for index in range(self.nh):
            self.Tj[index]=ran
        #initialising complex weight/threshold from hidden layer to output layer
        self.Wkj=np.zeros(6,dtype=complex)
        for j in range(self.nh):
           self.Wkj[l]=np.random.random()+np.random.random()*1j
        self.Tk=np.random.random()+np.random.random()*1j
    
    
    def train(self,Ii,T):
        
        Iib=np.conj(Ii)
        
        Hj=np.zeros(6,dtype=complex)
        for index in range(self.nh):
            var=Ii[0]*self.Wji[index]+self.Tj[index]
            Hj[index]=(1/(1+np.exp(-var.real)))+(1/(1+np.exp(-var.imag)))*1j
        sumvar=self.Tk
        
        for index in range(self.nh):
            sumvar=sumvar+Hj[index]*self.Wkj[index]    #calculating net for output layer
        O=(1/(1+np.exp(-sumvar.real)))+(1/(1+np.exp(-sumvar.imag)))*1j
       # print(O)
        E=T-O
        #Erms=0.5*np.square(np.abs(T-O)) #rms error of system
        LR=0.5  #Learning Rate
        dTk=LR*((E.real*(1-O.real)*(O.real))+(E.imag*(1-O.imag)*(O.imag))*1j)  #change in threshold of output layer
    
       
        Hjb=np.conj(Hj)  #array of complex conjugate of  output of hidden layer
        dwkj=Hjb*dTk
        dTj=np.zeros(6,dtype=complex)
        for index in range(self.nh):  #calculating change in bias/threshold value for neurons i hidden layer
            summation1=(E.real)*(1-O.real)*(O.real)*(self.Wkj[index].real)
            summation2=E.imag*(1-O.imag)*(O.imag)*(self.Wkj[index].imag)
            summation3=E.real*(1-O.real)*(O.real)*(self.Wkj[index].imag)
            summation4=E.imag*(1-O.imag)*(O.imag)*(self.Wkj[index].real)
            
            dTj[index]=LR*(((1-Hj[index].real)*(Hj[index].real)*(summation1+summation2))-((1-Hj[index].imag)*(Hj[index].imag)*(summation3-summation4))*1j)
        
        dwji=Iib*dTj
    
    
        self.Tj=self.Tj+dTj           #modifying threshold and weights
        self.Tk=self.Tk+dTk
        self.Wji=self.Wji+dwji
        self.Wkj=self.Wkj+dwkj
       
    def train_neurons(self,Inp): #noe=number of epochs
        for epoch in range(1001):
            for Ii in Inp:
                self.train(Ii[0],Ii[1])
       
    def query(self,Ii):
       
        Hj=np.zeros(6,dtype=complex)
        for index in range(self.nh):
            var=(Ii*self.Wji[index])+self.Tj[index]
            Hj[index]=(1/(1+np.exp(-var.real)))+(1/(1+np.exp(-var.imag)))*1j
        sumvar=self.Tk
        for index in range(self.nh):
            sumvar=sumvar+Hj[index]*self.Wkj[index]    #calculating net for output layer
        Op=(1/(1+np.exp(-sumvar.real)))+(1/(1+np.exp(-sumvar.imag)))*1j
        print(Op)
        return Op
   
    
x=[[[1+1j],[0.5+0.5j]],[[.1+.5j],[0.05+0.25j]],[[.8+.8j],[.4+.4j]],[[.7+.7j],[.35+.35j]]]
Net=Neural(x)
Net.train_neurons(x)

ai=1+0j
bi=0+1j
ci=np.sqrt(1/2)+np.sqrt(1/2)*1j


a=Net.query(ai)
b=Net.query(bi)
c=Net.query(ci)




circle1 = plt.Circle((0, 0), 1, color='r',fill=False)
circle2 = plt.Circle((0, 0), .5, color='b',fill=False)

ax = plt.gca()
ax.set_xlim((0, 2))
ax.set_ylim((0, 2))
ax.add_artist(circle1)
ax.add_artist(circle2)
plt.plot([0,1],[0,1],'k-') #plot a line from (0,0) to (1,1)
corr_outx=[ai.real/2,bi.real/2,ci.real/2]
corr_outy=[ai.imag/2,bi.imag/2,ci.imag/2]
nn_outx=[a.real,b.real,c.real]
nn_outy=[a.imag,b.imag,c.imag]

plt.plot(corr_outx,corr_outy,"o")
plt.plot(nn_outx,nn_outy,"o")


plt.show()



