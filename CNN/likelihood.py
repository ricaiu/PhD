import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import special


def energy_site(lattice,  x, y,  dim,  h):
	adjacent_sum=lattice[(x+1)%dim][y]+lattice[(x-1)%dim][y]+lattice[x][(y+1)%dim]+lattice[x][(y-1)%dim]
	return (lattice[x][y]*adjacent_sum+h*lattice[x][y])

def lattice_hamiltonian(lattice, dim,h):
    hamiltonian=0;
    for i in range(dim):
        for j in range(dim):
            hamiltonian= hamiltonian-energy_site(lattice, i, j, dim, h)/2+h*lattice[i][j]/2
    return hamiltonian



def canonical(size, index):
    arr = np.zeros(size)
    arr[index] = 1.0
    return arr


def hot_encodingT(y, N,T, Tmin, dn):
    for b in T:
        #print(b)
        for n in range(N):
            if ((b>=Tmin +n*dn) and (b<Tmin +(n+1)*dn)):
                hot_cod=canonical(N,n)
                #print(Tmin +n*dn)
        y.append(hot_cod)

def hot_encodingT_inverse(y,N,T,Tmin,dn):
    for i in range(len(y)):
        for j in range(len(y[i])):
            if y[i][j]:
                T.append(Tmin+j*dn)




def partition_func (L,beta):
	Z = 0
	for k in range(L**2+1):
		Z += scipy.special.binom(L**2,k)*np.exp((L**2-2*k)*beta)
	return Z

def likelihood_ising (lattice,b,L):
	Zl = partition_func(L,b)
	MinusBetaH = b*lattice_hamiltonian(lattice, L, 0)
	return np.exp(MinusBetaH)/Zl


Tstart = 0.1
Tend = 5.0
Nconf = 10**4
delta = (Tend - Tstart)/Nconf
N=100
dn=1/(N-2)
dnT = (Tend - Tstart)/N
L=16

Channel = 5
Nf = 3

T=[]
ide = np.identity(N)
hot_encodingT_inverse(ide,N,T,Tstart,dnT)
T=np.array(T)
print(T)



y_target = np.genfromtxt('/home/riccardo/DOTTORATO/CODE/CNN/cnn_data/y_test.txt' )

print(y_target.shape)

T_target=[]
hot_encodingT_inverse(y_target,N,T_target,Tstart,dnT)
T_target= np.array(T_target)
print(T_target.shape)

X = np.genfromtxt('/home/riccardo/DOTTORATO/CODE/CNN/cnn_data/x_test.txt' )
print(X.shape)

X = X.reshape(len(y_target),L,L)

minusH = lattice_hamiltonian(X[1],L,0)

print(minusH)

target_like = likelihood_ising(X[1], 1/T_target[1],L)

beta = [1/i for i in T]

ratio = [target_like/likelihood_ising(X[1],i,L) for i in beta ]

ratio = ratio/target_like





y_predict = np.genfromtxt('/home/riccardo/DOTTORATO/CODE/CNN/cnn_data/y_predict.txt' )


print(y_predict[1])



print(y_predict[1][y_target[1]==1]/y_predict[1])

plt.hist(y_predict[1][y_target[1]==1]/y_predict[1],bins=100)
plt.show()
