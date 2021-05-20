import numpy as np
import matplotlib.pylab as plt
import math

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

L=16
Tmin = 0.1
Tmax = 5.0
Nconf = 10**4
dt=(Tmax-Tmin)/Nconf

X=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/SW_LATT_cnn.dat' ,dtype='int32')
X=np.array(X)

y=int(len(X)/L**2)
X=np.reshape(X,(y,L,L))

M = []


for i in range(len(X)):
    m = X[i].sum()/(L**2)
    if m > 0:
        X[i] = - X[i]

    M.append(X[i].sum()/(L**2))

plt.plot(M)


x = np.arange(0.1,5.0,dt)

beta = np.array([1/i for i in x])
#x = np.array(np.linspace(0.,1.0,N))
x = np.array([truncate(i,2) for i in beta])[::-1]
nx = x.shape[0]
no_labels = 20# how many labels to see on axis x
step_x = int(nx / (no_labels - 1)) # step between consecutive labels

x_positions = np.arange(0,nx,step_x) # pixel count at label position
x_labels = x[::step_x] # labels you want to see
plt.xticks(x_positions, x_labels)

plt.show()
