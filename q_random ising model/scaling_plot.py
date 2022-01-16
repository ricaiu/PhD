from variables import *

def fit_func(x,a,b):
    return a*x+b


 

for sigma in sigmas:
    print('#######################')
 
    Tc[sigma] = np.load(sigma_path[sigma]+'/Tc.npy')
    nu = np.load(sigma_path[sigma]+'/nu.npy')
    two_min_eta = np.load(sigma_path[sigma]+'/2_min_eta.npy')
    
    print('SIGMA: ', sigma)
    print('Tc :', Tc[sigma])
    print('nu: ', nu)
    print('1/nu: ', 1/nu)
    print('2 - eta:', two_min_eta)

    for L in Ls[sigma]:
        print('L: ', L)          
        filename = 'L_'+str(L)
        tot_path = sigma_path[sigma]+filename

        


        m = np.load(tot_path+'/m.npy')
        dm = np.load(tot_path+'/dm.npy')

        chi = np.load(tot_path+'/chi.npy')
        dchi = np.load(tot_path+'/dchi.npy')

        binder = np.load(tot_path+'/binder.npy')
        dbinder = np.load(tot_path+'/dbinder.npy')


        if sigma<1:
            plt.figure(fig_index)
            plt.title('$U_L$ vs $tL^{1/v}$ at $\sigma$ = '+str(sigma))
            
            plt.xlabel('$tL^{1/v}$')
            plt.ylabel('$U_L$')        

            scaling = ((T[sigma][L]-Tc[sigma])/Tc[sigma])*L**(1/nu)
            
            plt.errorbar(scaling,binder,dbinder,label=str(L),marker = 'o')
            plt.legend()
            plt.figure(fig_index+1)
            plt.title('$\chi L^{-(2-\eta)}$ vs $tL^{1/v}$ at $\sigma$ = '+str(sigma))
            
            plt.xlabel('$tL^{1/v}$')
            plt.ylabel('$\chi L^{-(2-\eta)}$')        


            plt.errorbar(scaling,chi*L**(-two_min_eta),dchi*L**(-two_min_eta),label=str(L),marker = 'o')
            plt.legend()
            plt.figure(fig_index+2)
            plt.title('$m L^{((\eta-2)+1)/2}$ vs $tL^{1/v}$ at $\sigma$ = '+str(sigma))
            
            plt.xlabel('$tL^{1/v}$')
            plt.ylabel('$m L^{((\eta-2)+1)/2}$ ')        

            

            plt.errorbar(scaling,m*L**((1-two_min_eta)/2),dm*L**((1-two_min_eta)/2),label=str(L),marker = 'o')
            plt.legend()

    fig_index += 3
    print('#######################')



plt.show()

