import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('seaborn-paper')#('fivethirtyeight')#('seaborn-paper')

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




L=[32]


mode = 'Low'



if mode == 'Low':
    #temperature = ['01','02','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19' ,'20','21', '22', '23' ]
    temperature = np.arange(0.1,2.4,0.1)

    initial_temp = ['01', '05', '10','13','15','18','20']


if mode == 'High':

    temperature = [str(i) for i in np.arange(24,51,1)]
    #temperature = np.arange(24,51,1)

    initial_temp = ['24']#, '28','32','36','40','44','50']

temp = [truncate(int(i)/10,2) for i in temperature]

for l in L:


    for it in initial_temp:

        deltaT = [(i-int(it)/10) for i in temperature]

        start = int(it)/10
        x = np.genfromtxt('/home/riccardo/DOTTORATO/CODE/CNN/termo_data/'+str(l)+'/acc_termo_'+it+'.txt' )
        xx = np.genfromtxt('/home/riccardo/DOTTORATO/CODE/CNN/termo_data/'+str(l)+'/val_acc_termo_'+it+'.txt' )

        plt.figure(1)
        # plt.plot(x, marker ='o', linestyle = '--', label = 'train Tstart = ' + str(start))
        plt.plot(temperature,xx,marker ='o', linestyle = '--',label = 'val Tstart = ' + str(start)+'  L =' + str(l))
        plt.figure(2)
        #plt.plot(deltaT,x, marker ='o', linestyle = '--', label = 'train Tstart = ' + str(start))
        plt.plot(deltaT,(xx),marker ='', linestyle = '-',label = 'val Tstart = ' + str(start)+'  L =' + str(l))

plt.figure(1)
# x1 = np.array(temp)
# nx = x1.shape[0]
# no_labels = len(x1)# how many labels to see on axis x
# step_x = int(nx / (no_labels - 1)) # step between consecutive labels
#
# x_positions = np.arange(0,nx,step_x) # pixel count at label position
# x_labels = x1[::step_x] # labels you want to see
# plt.xticks(x_positions, x_labels)
plt.grid()
plt.xlabel('T')
plt.ylabel('Accuracy')
plt.title('Accuracy temperature difference between x axis and T on legend')
plt.legend(loc='best')

plt.figure(2)
plt.grid()
plt.xlabel('T')
plt.ylabel('Accuracy')
plt.title('Accuracy temperature difference')
plt.legend(loc='lower left')


plt.show()
