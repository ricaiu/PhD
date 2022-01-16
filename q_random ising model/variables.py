import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
from scipy.optimize import curve_fit


#Choose basic step: 'glauber' or 'metropolis'
mode = 'glauber'

#Number of tests
test =  {0.35 : 1,
           0.4 : 10,          
           0.75 : 10}


#If True the temperature scale with L to catch data near Tc
#If False the temeprature is the same for all Ls
scaling_temper = {0.35 : True,
           0.4 : True,          
           0.75 : False} 
#If scaling_temper lower_t and upper_t indicate the extremis of the temperature interval
# in the scaling form. Tc_file indicated the Tc guess for creating the interval

lower_t = {0.35 : -10,
           0.4 : -2,          
           0.75 : -20} 
upper_t = {0.35 : 10,
           0.4 : 2,
           0.75 : 20}
Tc_file = {0.35 : 1.6,
           0.4 : 2,
           0.75 : 1.78} 


#This is the Tc guess to extrapolate Tc in "tc_binder_ext"
Tc_guess = {0.35 : 1.6,
           0.4 : 2.01,
           0.75 : 1.78} 

#d is the spatial dimension of the generated lattice
#q is the number of neighbours
#sigmas contains the sigma parameters
#the power-law from which the neighbours are extracted is r**(-d+2-sigma)
d = 1
q = 3

sigmas = [0.4]

#num_T indicates the number of temperatures
num_T = {0.35 : 30,
           0.4 : 30,
           0.75 : 20} 


#Ls indicates the lenghts considered
Ls ={0.35 : np.array([256,512,1024,2048,4096,8192,16384]),
    0.4 : np.array([1024,2048,4096,8192]),
    0.75 : np.array([256,512,1024,2048,4096,8192]),
    1.2 : np.array([256,512,1024,2048,4096,8192,16384,32768]) }



#In T are stored the temperatures
#If scaling_temper they are then replaced with the correct arrays
T_035 = np.linspace(1,2,num_T[0.35])
T_04 = np.linspace(1.4,2.4,num_T[0.4])
T_075 = np.linspace(1.65,2,num_T[0.75])
T_12 = np.linspace(1,2,30)


T = {0.35 : T_035,
     0.4 : T_04,
     0.75: T_075,
     1.2 : T_12}
#Make the correct T arrays
for sigma in sigmas:
    temper = np.empty((len(Ls[sigma]),num_T[sigma]))
    if scaling_temper[sigma]:
        
        for i in range(len(Ls[sigma])):
          temper[i] = np.linspace(Tc_file[sigma]*(1+lower_t[sigma]*(Ls[sigma][i])**(-sigma)),
                                Tc_file[sigma]*(1+upper_t[sigma]*(Ls[sigma][i])**(-sigma)),num_T[sigma])
          
        T[sigma] = dict(zip(Ls[sigma], temper))
    else:
      temper[:] = T[sigma]
      T[sigma] = dict(zip(Ls[sigma], temper))  



#utilities
#this are utilities for plotting and collecting data
fig_index = 1
tmp = ['' for i in range(len(sigmas))]

sigma_path = dict(zip(sigmas,tmp))

for sigma in sigmas:
    tmp = mode+'/1D/sigma'+str(sigma)+'/'
    if scaling_temper[sigma]:
        #test_path = 'q'+str(q)+'scale_temp_min'+str(lower_t[sigma])+'_max'+str(upper_t[sigma])+'_nT'+str(num_T[sigma])+'/'
        test_path = ('q'+str(q)+'scale_temp_Tc'+str(Tc_file[sigma])+'_min'+str(lower_t[sigma])+
                  '_max'+str(upper_t[sigma])+'_nT'+str(num_T[sigma])+'/')
    else:
        test_path = 'q'+str(q)+'_Tmin'+str(T[sigma][Ls[sigma][0]][0])+'_Tmax'+str(T[sigma][Ls[sigma][0]][-1])+'_nT'+str(num_T[sigma])+'/'

    sigma_path[sigma] = tmp+test_path


Tc = {0.35 : 0,
      0.4 : 0,
      0.75: 0}