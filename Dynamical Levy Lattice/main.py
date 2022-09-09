"""

# Dynamical Levy Lattice
---
1) Choose a random spin $S_i$  
2) Draw $q$ neighbours $S_{r_j}$, each at distance $r_j$ from $S_i$, using as probability distribution $P(r)\sim r^{-(d+\sigma)}$  
3) Calculate the energy difference $\Delta E = 2\cdot J \cdot S_i \sum_{j=1}^q  S_{r_j}$  
4) Flip $S_i$ following the wanted prescription (Glauber or Metropolis)  
---

Import parameters and functions
"""

from parameters import *
from functions import *


"""Print in an output file the chosen parameters"""

with open(Parameters['Output file name'] +'.txt', 'a') as f:
  print('START SIMULATION', file = f)
  print('PARAMETERS:', file = f)
  for keys in Parameters:
    print(keys, ' : ', Parameters[keys], file = f)

"""Make a "Ts.npy" file where are stored temperatures for thermodynamics analysis and a file "Ts.txt" which could be useful for future simulations"""

make_Temperature_file(Parameters['Sizes'],Parameters['Path'],
                      Parameters['Temperatures'], Parameters['Steps'])

"""Create an empty array where will be stored temporarily the $q$ neighbours, to speed up the single spin-flip"""

neighs = np.empty((Parameters['Neighbours'],Parameters['Dimension']),dtype = 'int16')

"""Create an array of possibile energy interaction, in order to speed up the calculation of spin-flip energy  
For example if $q = 3$, then E = [-3,-1,1,3] for Glauber and E = [1,3] for Metropolis
"""

energy = np.array(create_energy(Parameters['Neighbours'],[-1,1],Parameters['Prescription']))
print(energy)

"""Choosing the right requested dynamics"""

dynamics = choose_dynamics(Parameters)

"""Start Simulation!"""

for L in Parameters['Sizes']:

  #Make a directory named after the size "L_{size}" where store the observable
  try:
      os.mkdir(Parameters['Path']+'L_'+str(L))
  except OSError as error:
      pass
  saving_directory = Parameters['Path']+'L_'+str(L)+'/'

  #Take the temperatures for readability
  T = Parameters['Temperatures'][L]

  #Set termalization steps
  termalization = Parameters['Autocorrelation']*L**Parameters['Dimension']

  #Create the lattice
  if Parameters['Dimension'] == 1:
    shape = (L)
  elif Parameters['Dimension'] == 2:
    shape = (L,L)
  if Parameters['Start'] == 'hot':
    lattice = 2*np.round(np.random.random(shape))-1
  elif Parameters['Start'] == 'cold':
    lattice = np.ones(shape,dtype = 'int16')

  #Make lattice of integers for memory resource
  lattice = lattice.astype( 'int16')
  
  #Set observables (here the magnetizaion, i.e. the order parameter)

  magnetization = np.empty((len(T),Parameters['Steps']))



  #Temperature has to be inverted if 'hot' conditions are requested
  if Parameters['Start'] == 'hot':
    T = T[::-1]

  #Set look-up table of perscription's filters for spin-flip energy

  if Parameters['Prescription'] == 'metropolis':
    filt = [np.exp(-2*(1/T[i])*energy) for i in range(len(T))]
  elif Parameters['Prescription'] == 'glauber':
    filt = [1/(1+ np.exp(2*(1/T[i])*energy))for i in range(len(T))]

#Now start the dynamics for each Temperature

  #Track the temperature index with tmp_index
  tmp_index = 0
  for t in T:
    #save time to record the time consuming
    now_time = time.time()

    #Termalize the system
    for i in range(termalization):
      dynamics(lattice,Parameters['Neighbours'],
                     Parameters['Dimension']+Parameters['Sigma'],
                     neighs,filt[tmp_index],energy)
      
    for k in range(Parameters['Steps']):     
      for i in range(L**Parameters['Dimension']):           
        dynamics(lattice,Parameters['Neighbours'],
                    Parameters['Dimension']+Parameters['Sigma'],
                    neighs,filt[tmp_index],energy)     

      magnetization[tmp_index,k]= lattice.sum()/L**Parameters['Dimension']

      #Saving the magnetization during the simulation, in order to get results
      #even if the simulation stops before it ends
      if Parameters['Start'] == 'hot':
        np.save(saving_directory+'/magnetization.npy',magnetization[-tmp_index-1:])
      else:
        np.save(saving_directory+'/magnetization.npy',magnetization[:tmp_index+1])
    
    #Print time consuming for Parameters['Steps'] MCS:
    print('T= ',t,"--- %s seconds ---" % (time.time()-now_time))
    with open(Parameters['Output file name']+'.txt', 'a') as f:
      print('T= ',t,"--- %s seconds ---" % (time.time()-now_time),file = f ) 
    
    tmp_index += 1

  #Saving the total magnetization. Magnetization is stored from the lowest
  #temperature to the highest, also if the temperature is lowered in the dynamics
  if Parameters['Start'] == 'hot':
    magnetization = magnetization[::-1]
  np.save(saving_directory+'/magnetization.npy',magnetization)
