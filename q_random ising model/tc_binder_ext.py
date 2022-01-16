from variables import *

def fit_func(x,a,b):
    return a*x+b

num_pts = 10**4
threesh = 10**-1 #threesh distance between binders at L and 2L considered as crossing point



for sigma in sigmas:
    print('#######################')

    
    colors = cm.rainbow(np.linspace(0, 1, len(Ls[sigma])))
    print('SIGMA: ', sigma)

    x_plot = np.empty((len(Ls[sigma]),num_pts))
    y_plot = np.empty((len(Ls[sigma]),num_pts))
    Tc_rect = np.empty(len(Ls[sigma])-1)
    L_idx = 0

    for L,clr in zip(Ls[sigma], colors):
        print('L: ', L)

 
        filename = 'L_'+str(L)
        tot_path = sigma_path[sigma]+filename

        binder = np.load(tot_path+'/binder.npy')
        dbinder = np.load(tot_path+'/dbinder.npy')

        binder_rect = np.empty(5)
        dbinder_rect = np.empty(5)
        T_rect = np.empty(5)
        binder_rect[:3] = binder[T[sigma][L] < Tc_guess[sigma]][-3:]
        binder_rect[-2:] = binder[T[sigma][L] > Tc_guess[sigma]][:2]
        dbinder_rect[:3] = dbinder[T[sigma][L] < Tc_guess[sigma]][-3:]
        dbinder_rect[-2:] = dbinder[T[sigma][L] > Tc_guess[sigma]][:2]
        T_rect[:3] = T[sigma][L][T[sigma][L] < Tc_guess[sigma]][-3:]
        T_rect[-2:] = T[sigma][L][T[sigma][L] > Tc_guess[sigma]][:2]


        popt, pcov = curve_fit(fit_func,T_rect,binder_rect,sigma = dbinder_rect,p0=[-1,0])
        perr = np.sqrt(np.diag(pcov))

        x_plot[L_idx] = np.linspace(T_rect[0]-0.01, T_rect[-1]+0.01,num_pts)
        y_plot[L_idx] = fit_func(x_plot[L_idx],popt[0],popt[1])

        plt.figure(fig_index)
        plt.title('$T_c^*$ crossing point via linear fitting at $\sigma=$ '+str(sigma))
        plt.xlabel('$T$')
        plt.ylabel('$U_2$')
        plt.plot(T[sigma][L],binder, linestyle = '-',marker = '.', color= 'grey', alpha = 0.3)
        plt.errorbar(T_rect,binder_rect,dbinder_rect, linestyle='',
                        marker = 'o', color = clr)
        plt.plot(x_plot[L_idx],y_plot[L_idx],label = str(L), color = clr)
        plt.legend()

        L_idx += 1

    for i in range(len(Ls[sigma])-1):
        x_idx = int(len(x_plot[i][abs(y_plot[i] - y_plot[i+1])<threesh])/2)
        Tc_rect[i] = x_plot[i][abs(y_plot[i] - y_plot[i+1])<threesh][x_idx]


    plt.figure(fig_index +1)  
    plt.title('1/L vs $T_c^*$ at $\sigma=$ '+str(sigma)) 
    plt.xlabel('1/L')
    plt.ylabel('$T_c^*$')
    plt.plot(1/Ls[sigma][1:],Tc_rect,linestyle = '', marker = 'o')
    popt, pcov = curve_fit(fit_func,1/Ls[sigma][1:],Tc_rect,p0=[-1,Tc_guess[sigma]])
    perr = np.sqrt(np.diag(pcov))
    l_plot = np.linspace(0,1/Ls[sigma][1],10000)
    plt.plot(l_plot,popt[1]+popt[0]*l_plot,label='curve_fit')
    print('Tc extrapolated: ', popt[1],' +/- ', perr[1])
    #print('m extrapolated: ', popt[0],' +/- ', perr[0]) 
    np.save(sigma_path[sigma]+'/Tc.npy',popt[1]) 

    fig_index += 2
    print('#######################')

plt.show()



