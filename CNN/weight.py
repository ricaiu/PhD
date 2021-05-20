import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import lmfit
from scipy.optimize import curve_fit
from lmfit.models import StepModel, LinearModel




def fitfunc(a,b,c,tc,t):
    return a*np.tanh(b*(t-tc))-c
def canonical(size, index):
    arr = np.zeros(size)
    arr[index] = 1.0
    return arr


def hot_encoding(y_label, N, beta):
    for b in beta:
        if (b<0):
            hot_cod = canonical(N,0)
        elif(b>=1):
            hot_cod = canonical(N,N-1)
        else:
            for n in range(N):
                if ((b>=n*dn) and (b<(n+1)*dn)):
                    hot_cod=canonical(N,n+1)
        y_label.append(hot_cod)
    y_label = np.reshape(y_label,(len(T),N))

def hot_encodingT(y, N,T, Tmin, Tmax,dn):
    for b in T:
        #print(b)
        for n in range(N):
            if ((b>=Tmin +n*dn) and (b<Tmin +(n+1)*dn)):
                hot_cod=canonical(N,n)
                #print(Tmin +n*dn)
        y.append(hot_cod)


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



mode = 'T' #choose between 'beta02'  'beta'  'T'


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

T = []
y_label = []
T_fromfile = np.genfromtxt('/home/riccardo/DOTTORATO/CODE/SW/provaT_cnn.txt' )
for i in np.arange(0,len(T_fromfile),3):
    T.append(T_fromfile[i])

T = np.array(T)

beta = np.array([1/i for i in T])





# for ind in np.arange(8000,10000,50):
#     print(y_label[ind], beta[ind])




X=  np.genfromtxt('/home/riccardo/DOTTORATO/CODE/CNN/cnn_data/weight_matrix.txt' )





hot_encodingT(y_label,N,T,Tstart,Tend,dnT)

y_label = np.array(y_label)



plt.figure(1)
plt.imshow(X, cmap ='Greys',aspect='auto' )
plt.colorbar()


filters=  np.genfromtxt('/home/riccardo/DOTTORATO/CODE/CNN/cnn_data/filter_matrix.txt' )

filters = filters.reshape((int(len(filters)/Nf),Nf,Nf))
print(filters.shape)

print(X.shape)
print(y_label[0].shape)



# # normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)
if mode == 'beta0.2':
    x = np.arange(beta[-1],1.0,dn)
if mode == 'beta':
    x = np.arange(0.,1.0,dn)
if mode == 'T':
    x = np.arange(Tstart,Tend,dnT)
#print(x_test)
#x = np.arange(0,1.0,dn)
# the grid to which your data corresponds
#x = np.array(np.linspace(0.,1.0,N))
x = np.array([truncate(i,2) for i in x])
nx = x.shape[0]
no_labels = 20# how many labels to see on axis x
step_x = int(nx / (no_labels - 1)) # step between consecutive labels

x_positions = np.arange(0,nx,step_x) # pixel count at label position
x_labels = x[::step_x] # labels you want to see
plt.xticks(x_positions, x_labels)
# in principle you can do the same for y, but it is not necessary in your case

#print(X.T[N-1].shape)

C = []
for i in range(N):
    #C.append(X[:,i].dot(y_label[i]))
    C.append(X.T[i].sum())
C = np.array(C)
print(C.shape)


dC = [0.5 for i in C]
print(dC)


ig = [5.,6.,5.,1.]

# model data as Step + Line
step_mod = StepModel(form='linear', prefix='step_')
line_mod = LinearModel(prefix='line_')

model = step_mod + line_mod

# make named parameters, giving initial values:
pars = model.make_params(line_intercept=C.min(),
                         line_slope=0,
                         step_center=x.mean(),
                         step_amplitude=C.std(),
                         step_sigma=2.0)

# fit data to this model with these parameters
out = model.fit(C, pars, x=x)

# print results
print(out.fit_report())

# plot data and best-fit
plt.figure(2)
plt.plot(x, C, linestyle='', marker = '+', color = 'b')
plt.plot(x, out.best_fit, 'r-')






# popt, pcov = curve_fit(fitfunc,x,C,sigma = dC, p0 =ig)
#
# print(popt,pcov)
# x_plot = np.arange(0.1,5.0,0.001)
# C_plot = fitfunc(popt[0],popt[1],popt[2],popt[3],x_plot)
# #C_plot = fitfunc(5,7,5,2,x_plot)
#
# plt.figure(2)
# plt.errorbar(x,C,dC, linestyle ='', marker = '.')
# plt.plot(x_plot,C_plot, linestyle ='-', marker = '')



plt.figure(3)



n_filters, ix = Channel, 1
for i in range(n_filters):

    # get the filter
    f = filters[i]
    print('sum of F: ',f.sum())
    print('mean of F: ',f.mean())
    # plot each channel separately

    # specify subplot and turn of axis
    ax = plt.subplot(n_filters, 3, ix)
    ax.set_xticks([])
    ax.set_yticks([])
    # plot filter channel in grayscale
    plt.imshow(f,cmap ='coolwarm' )
    plt.colorbar()
    ix += 1



plt.show()
