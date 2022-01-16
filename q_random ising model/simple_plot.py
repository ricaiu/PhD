from variables import *


fig_index = 1
for sigma in sigmas:

    print('SIGMA: ', sigma)

    max_chi = []
    dmax_chi = []
    binder_tc = []
    dbinder_tc = []
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

        max_chi.append(max(chi))

        dmax_chi.append(dchi[chi.argmax()])

        plt.figure(fig_index)
        plt.title('$\chi$ vs T at $\sigma$ = '+str(sigma))
        plt.xlabel('T')
        plt.ylabel('$\chi$')
        

        plt.errorbar(T[sigma][L],chi,dchi,label = str(L),marker = 'o')

        plt.legend()


        plt.figure(fig_index+1)
        plt.title('|m| vs T at $\sigma$ = '+str(sigma))
        plt.xlabel('T')
        plt.ylabel('|m|')        


        plt.errorbar(T[sigma][L],m,dm,label=str(L),marker = 'o')
        plt.legend()

        plt.figure(fig_index+2)
        plt.title('$U_L$ vs T at $\sigma$ = '+str(sigma))
        plt.xlabel('T')
        plt.ylabel('$U_L$')
        

        plt.errorbar(T[sigma][L],binder,dbinder,label = str(L),marker = 'o')

        plt.legend()  

    fig_index += 3







    plt.figure(fig_index)
    plt.title('$\chi$ vs $L$ at $\sigma$ = '+str(sigma))
    plt.xlabel('L')
    plt.ylabel('$\chi$')    

    plt.errorbar(Ls[sigma],max_chi,marker = 'o',label='experiment' )

    plt.xscale('log')
    plt.yscale('log')




    fig_index += 1


plt.show()

