from variables import *


def rect(x,a,b):
    return a*x+b
def parab(x,a,b,c):
    return a*x**2+b*x+c


for sigma in sigmas:
  print('#######################')

  colors = cm.rainbow(np.linspace(0, 1, len(Ls[sigma])))
  print('SIGMA: ', sigma)


  T_max = np.empty(len(Ls[sigma]))
  chi_max = np.empty(len(Ls[sigma]))
  L_idx = 0
  for L,clr in zip(Ls[sigma], colors):
      print('L: ',L)

      filename = 'L_'+str(L)
      tot_path = sigma_path[sigma]+filename

      chi = np.load(tot_path+'/chi.npy')
      dchi = np.load(tot_path+'/dchi.npy')
      max_chi = max(chi)
      chi_parab = chi[abs(chi-max_chi)/max_chi < 0.2]
      dchi_parab = dchi[abs(chi-max_chi)/max_chi < 0.2]
      T_parab = T[sigma][L][abs(chi-max_chi)/max_chi < 0.2]
      print(len(T_parab),len(chi_parab))

  
      plt.figure(fig_index)
      plt.title('$T_M$ and $\chi_M$ via parable fitting at $\sigma=$ '+str(sigma))
      plt.plot(T[sigma][L],chi, marker = '.',color= 'grey',alpha = 0.4)
      
      plt.figure(fig_index)
      plt.plot(T_parab, chi_parab, marker = 'o', label = str(L),color = clr)
      plt.legend()
      popt, pcov = curve_fit(parab,T_parab,chi_parab,sigma = dchi_parab,p0=[0,0,0]) 
      perr = np.sqrt(np.diag(pcov))
      T_max[L_idx] = -popt[1]/(2*popt[0])
      chi_max[L_idx] = popt[2]-(popt[1]**2)/(4*popt[0])

      x_plot  = np.linspace(T_parab[0],T_parab[-1], 10**4 )
      y_plot = parab(x_plot,popt[0],popt[1],popt[2])
      plt.plot(x_plot,y_plot,color = clr)
      
      plt.xlabel('$T$')
      plt.ylabel('$\chi$') 


      L_idx += 1

  plt.plot(T_max,chi_max,linestyle = '',marker = 'o',color='red')
  plt.legend()
  
  Tc = np.load(sigma_path[sigma]+'/Tc.npy')

  popt, pcov = curve_fit(rect,np.log(Ls[sigma]),np.log(abs(T_max-Tc)), p0=[1,1])
  perr = np.sqrt(np.diag(pcov))
  print('1/nu = ', -popt[0],' +/- ', perr[0])
  print('nu = ', -1/popt[0],' +/- ', perr[0]/popt[0]**2)
  np.save(sigma_path[sigma]+'/nu.npy',-1/popt[0])

  plt.figure(fig_index+1)
  plt.title('$L$ vs $T_M-T_c$ at $\sigma=$ '+str(sigma))
  plt.plot(Ls[sigma],abs(T_max-Tc), linestyle = '', marker = 'o' )

  x_plot = np.linspace(Ls[sigma][0]/2,Ls[sigma][-1]*2,10000)
  plt.plot(x_plot,np.exp(popt[1])*x_plot**popt[0],label='curve_fit')
  plt.xlabel('$L$')
  plt.ylabel('$T_M-T_c$')   
  plt.xscale('log')
  plt.yscale('log')
  plt.legend()

  popt, pcov = curve_fit(rect,np.log(Ls[sigma]),np.log(chi_max),p0=[sigma,0])#
  perr = np.sqrt(np.diag(pcov))
  print('2-eta (',sigma,'): ',popt[0],'+/-',perr[0])
  np.save(sigma_path[sigma]+'/2_min_eta.npy',popt[0])

  plt.figure(fig_index+2)
  plt.title('$\chi$ vs $L$ at $\sigma$ = '+str(sigma))
  plt.xlabel('L')
  plt.ylabel('$\chi$')    

  plt.errorbar(Ls[sigma],chi_max,marker = 'o',label='experiment' )


  x_plot = np.linspace(Ls[sigma][0]/2,Ls[sigma][-1]*2,10000)
  plt.plot(x_plot,np.exp(popt[1])*x_plot**popt[0],label='curve_fit')
  plt.plot(x_plot,np.exp(popt[1])*x_plot**sigma,label='$L^{\sigma}$')
      

  plt.xscale('log')
  plt.yscale('log')
  plt.legend()


  fig_index += 3
  print('#######################')

plt.show()
