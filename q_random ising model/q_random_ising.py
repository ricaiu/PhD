import numpy as np
import time
import os
import random

def q_metropolis_1D(lattice,q,a,neighs,filt):
  L = len(lattice)
 
  center = int(L*random.random())%L#sampling the starting site
   
  for k in range(q):
    radius = ((1 + ((2/L)**a-1)*random.random())**(-1/a))
    neighs[k] =int(radius*(2*round(random.random())-1))
   
  if neighs[0] == neighs[1]:
    neighs[1] *= -1
    while neighs[0] == neighs[2] or neighs[1] == neighs[2]:
      radius = ((1 + ((2/L)**a-1)*random.random())**(-1/a))
      neighs[2] =int(radius*(2*round(random.random())-1))
      
  elif neighs[0] == neighs[2]:
    neighs[2] *= -1
    while neighs[0] == neighs[1] or neighs[2] == neighs[1]:
      radius = ((1 + ((2/L)**a-1)*random.random())**(-1/a))
      neighs[1] =int(radius*(2*round(random.random())-1))

  neighs_sum = 0
  for i in range(q):
    neighs_sum+= lattice[(center+int(neighs[i]))%L]
  energy = lattice[center]*neighs_sum #energy of the interaction neighs-center
  if energy < 0:
    lattice[center]*=-1
  else:
    probability = random.random()
    if energy == 3:
      if (filt[1] > probability ): 
         lattice[center]*=-1
    else:
      if (filt[0] > probability ): 
         lattice[center]*=-1   

def q_glauber_1D(lattice,q,a,neighs,filt):
  L = len(lattice)
 
  center = int(L*random.random())%L#sampling the starting site
   
  for k in range(q):
    radius = ((1 + ((2/L)**a-1)*random.random())**(-1/a))
    neighs[k] =int(radius*(2*round(random.random())-1))
   
  if neighs[0] == neighs[1]:
    neighs[1] *= -1
    while neighs[0] == neighs[2] or neighs[1] == neighs[2]:
      radius = ((1 + ((2/L)**a-1)*random.random())**(-1/a))
      neighs[2] =int(radius*(2*round(random.random())-1))
      
  elif neighs[0] == neighs[2]:
    neighs[2] *= -1
    while neighs[0] == neighs[1] or neighs[2] == neighs[1]:
      radius = ((1 + ((2/L)**a-1)*random.random())**(-1/a))
      neighs[1] =int(radius*(2*round(random.random())-1))

  neighs_sum = 0
  for i in range(q):
    neighs_sum+= lattice[(center+int(neighs[i]))%L]
  energy = lattice[center]*neighs_sum #energy of the interaction neighs-center
  probability = random.random()
  if energy == -3:
    if probability < filt[0]:
     lattice[center]*=-1
  elif energy == -1:
    if probability < filt[1]:
     lattice[center]*=-1
  elif energy == 1:
    if probability < filt[2]:
     lattice[center]*=-1
  else:
    if probability < filt[3]:
     lattice[center]*=-1


def q_metropolis_1D_SR_structure(lattice,q,a, neighs,direction,radius,filt):
  L = len(lattice)

  
  center = round(L*random.random())%L#sampling the starting site
  
  
  for k in range(q):  
    direction[k] = 2*round(random.random())-1
    radius[k] = (-(random.random()*((L/2)**a-(2)**a)-(L/2)**a)/(L**a))**(-1/a)
    neighs[k] =int(radius[k]*direction[k])
    
  while neighs[0] == neighs[1] or neighs[0] == neighs[2] or neighs[2] == neighs[1]:
    for k in range(q):
      radius[k] = (-(random.random()*((L/2)**a-(2)**a)-(L/2)**a)/(L**a))**(-1/a)
      neighs[k] =int(radius[k]*direction[k])
    
  
  neighs_sum = 0
  for i in range(q):
    neighs_sum+= lattice[(center+int(neighs[i]))%L]

  neighs_sum+=lattice[(center+1)%L]+lattice[(center-1)%L]


  energy = lattice[center]*neighs_sum #energy of the interaction neighs-center


  if energy <= 0:
    lattice[center]*=-1
  else:
    probability = random.random()
    if energy == 3:
      if (filt[1] > probability ): 
         lattice[center]*=-1
    else:
      if (filt[0] > probability ): 
         lattice[center]*=-1   

 

#GLAUBER OR METROPOLIS?
mode = 'glauber'

test = 0

#PARAMETERS
scaling_temper = False
lower_t = -20
upper_t = 20
Tc = 1.6

Ls = np.array([256,512,1024,2048,4096])

metro_time = 0.7*10**-5

steps = 1000



T = np.linspace(1.4,2.4,20)

d = 1
sigma = 0.4
q = 3
eta = 2*steps*Ls*metro_time*len(T)

eta = eta.sum()
print('Estimated time: ',eta/3600)



with open('outs/out'+str(sigma)+'.txt', 'a') as f:
  print('Estimated time: ',eta/3600, file = f)





neighs_termo = np.empty(q)

if mode == 'metropolis':
  energy = np.array([1,3])
  filt = [np.exp(-2*(1/T[i])*energy) for i in range(len(T))]
elif mode == 'glauber':
  energy = np.array([-3,-1,1,3])
  filt = [1/(1+ np.exp(2*(1/T[i])*energy))for i in range(len(T))]





for L in Ls:
  
  termalization = steps*L

  lattice = 2*np.round(np.random.rand(L))-1#np.ones(L)
  

  sigma_path = mode+'/1D/sigma'+str(sigma)+'/'
  test_path = 'q'+str(q)+'_Tmin'+str(T[0])+'_Tmax'+str(T[-1])+'_nT'+str(len(T))+'/'
  if scaling_temper:
    test_path = ('q'+str(q)+'scale_temp_Tc'+str(Tc)+'_min'+str(lower_t)+
                  '_max'+str(upper_t)+'_nT'+str(len(T))+'/')
  filename = 'L_'+str(L)
  tot_path = sigma_path+test_path+filename
  print(tot_path)
  print('estimated time: ',2*(metro_time)*L*steps*len(T), ' seconds  ',2*(metro_time)*L*steps*len(T)/60,' minutes ' ,2*(metro_time)*L*steps*len(T)/3600, ' hours' )
  with open('outs/out'+str(sigma)+'.txt', 'a') as f:
    print(tot_path,file = f)
    print('estimated time: ',2*(metro_time)*L*steps*len(T), ' seconds  ',2*(metro_time)*L*steps*len(T)/60,' minutes ' ,2*(metro_time)*L*steps*len(T)/3600, ' hours',file = f )
  
  if not os.path.isdir(sigma_path):
    os.mkdir(sigma_path)
  
  if not os.path.isdir(sigma_path+test_path):
    os.mkdir(sigma_path+test_path)
  if not os.path.isdir(tot_path):
    os.mkdir(tot_path)  
  m = np.empty((len(T),steps))

  tmp_index = 0

  if scaling_temper:
    T = np.linspace(Tc*(1+lower_t*L**(-sigma)),Tc*(1+upper_t*L**(-sigma)),len(T))
    if mode == 'metropolis':
      filt = [np.exp(-2*(1/T[i])*energy) for i in range(len(T))]
    else:
      filt = [1/(1+ np.exp(2*(1/T[i])*energy))for i in range(len(T))]

  for b in T:
    now_time = time.time()   
    for i in range(termalization):
      if mode == 'metropolis':
        q_metropolis_1D(lattice,q,sigma,neighs_termo,filt[tmp_index])
      elif mode == 'glauber':
        q_glauber_1D(lattice,q,sigma,neighs_termo,filt[tmp_index]) 
    for k in range(steps):     
      for i in range(L):           
        if mode == 'metropolis':
          q_metropolis_1D(lattice,q,sigma,neighs_termo,filt[tmp_index])
        elif mode == 'glauber':
          q_glauber_1D(lattice,q,sigma,neighs_termo,filt[tmp_index])      

      m[tmp_index,k]= lattice.sum()/L

    print(b,"--- %s seconds ---" % (time.time()-now_time))
    with open('outs/out'+str(sigma)+'.txt', 'a') as f:
      print(b,"--- %s seconds ---" % (time.time()-now_time),file = f )
    
    tmp_index += 1
  np.save(tot_path+'/m_exp'+str(test)+'.npy',m)
