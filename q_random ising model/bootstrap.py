from variables import *

def bootstrap(func,sample,num_of_estim):
  N = len(sample)
  M = num_of_estim
  estimators = np.empty(M)
  for i in range(M):   
    #sampling N indexes 
    indexes = np.random.randint(0,N,size = N)
    new_sample= sample[indexes]
    estimators[i] = func(new_sample)

  return estimators.std()

def binder(sample):
  x2 = (sample**2).mean()
  x4 = (sample**4).mean()
  return 1-x4/(3*x2**2)


def susce(sample):
  x = abs(sample).mean()
  x2 = (sample**2).mean()
  return (x2-x**2)

#PARAMETERS
num_of_estim = 100


for sigma in sigmas:

    m = np.empty(num_T[sigma])
    dm = np.empty(num_T[sigma])
    binders = np.empty(num_T[sigma])
    dbinders = np.empty(num_T[sigma])
    chi = np.empty(num_T[sigma])
    dchi = np.empty(num_T[sigma])

    for L in Ls[sigma]:

  
        filename = 'L_'+str(L)
        tot_path = sigma_path[sigma]+filename


        ms = np.load(tot_path+'/m_exp0.npy')
        for i in range(1,test[sigma]):
          tmp = np.load(tot_path+'/m_exp'+str(i)+'.npy')
          ms = np.append(ms,tmp,axis = 1)
        print('shape of data: ', ms.shape)
        now_time = time.time()

        for t in range(num_T[sigma]):
            sample = ms[t]
            m[t] = abs(sample).mean()
            dm[t] = abs(sample).std()
            binders[t] = binder(sample)
            dbinders[t] = bootstrap(binder,sample,num_of_estim)
            chi[t] = susce(sample)*(L/T[sigma][L][t])
            dchi[t] = bootstrap(susce,sample,num_of_estim)*(L/T[sigma][L][t])


        np.save(tot_path+'/m.npy',m)
        np.save(tot_path+'/dm.npy',dm)
        np.save(tot_path+'/binder.npy',binders)
        np.save(tot_path+'/dbinder.npy',dbinders)
        np.save(tot_path+'/chi.npy',chi)
        np.save(tot_path+'/dchi.npy',dchi)

        print('sigma: ', sigma, ' L: ', L,' time: ', time.time() - now_time)
