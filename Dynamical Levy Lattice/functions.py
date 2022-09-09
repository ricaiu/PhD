import numpy as np
import itertools
import time
import random
import os

def define_limits_2D(L):
  '''
  Defines the upper and lower bound of the pareto distribution, dim = 2
  '''
  UPPER = L/np.sqrt(2)-1.000001/np.sqrt(2)
  LOWER = 1.000001/np.sqrt(2)
  return UPPER, LOWER 

def define_limits_1D(L):
  '''
  Defines the upper and lower bound of the pareto distribution, dim = 1
  '''
  UPPER = int(L/2)+0.9999999
  LOWER = 1
  return UPPER, LOWER 

def generalized_uniform(unif,upper,lower):
  '''
  return a variable uniform distributed in [lower,upper]
  '''
  return unif*(upper-lower)+lower

def generalized_pareto(unif,upper,lower,alpha):
  '''
  return a variable distributed following r^-alpha in [lower,upper]
  '''
  return (-(unif*upper**(alpha-1)-unif*lower**(alpha-1)-upper**(alpha-1))/((upper*lower)**(alpha-1)))**(-1/(alpha-1))

def is_in(coordinate,L):
  '''
  Boolean for coordinate in [0,L-1]
  '''
  if coordinate <= L-1 and coordinate >=0:
    return True
  
  return False

def sampling_starting_spin_linear(L):
  '''
  Sample the starting spin, dim = 1
  '''
  return int(L*random.random())%L

def sampling_starting_spin_quadratic(L):
  '''
  Sample the starting spin, dim = 2
  '''
  return int(L*random.random())%L,int(L*random.random())%L

def sampling_q_neighbours_linear_bounded(q,neighs,UPPER,LOWER,alpha):
  '''
  Sample q neighbour using the Bounded Pareto with parameter alpha in dim = 1
  '''
  for k in range(q):
    radius = generalized_pareto(random.random(),UPPER,LOWER,alpha)
    neighs[k] =int(radius*(2*round(random.random())-1)) 

def sampling_q_neighbours_quadratic_bounded(q,neighs,UPPER,LOWER,alpha):
  '''
  Sample q neighbour using the Bounded Pareto with parameter alpha in dim = 2
  '''
  for k in range(q):
    radius = generalized_pareto(random.random(),UPPER,LOWER,alpha)
    theta = generalized_uniform(random.random(),2*np.pi,0)
    neighs[k,0] = radius*np.cos(theta)
    neighs[k,1] = radius*np.sin(theta)

def sampling_q_neighbours_quadratic_infinite(q,neighs,LOWER,alpha):
  '''
  Sample q neighbour using the Bounded Pareto with parameter alpha, UPPER = inf
  in dim = 2
  '''
  for k in range(q):
    radius = ((1-random.random())**(-1/(alpha-1)))*LOWER 
    theta = generalized_uniform(random.random(),2*np.pi,0)
    neighs[k,0] = radius*np.cos(theta)
    neighs[k,1] = radius*np.sin(theta)
    
def sampling_q_neighbours_linear_infinite(q,neighs,LOWER,alpha):
  '''
  Sample q neighbour using the Bounded Pareto with parameter alpha, UPPER = inf
  in dim = 1
  '''
  for k in range(q):
    radius = ((1-random.random())**(-1/(alpha-1)))*LOWER 
    neighs[k] =int(radius*(2*round(random.random())-1))

def interaction_energy_2D_PBC(q,L,lattice,neighs,center_x,center_y):
  '''
  Compute the interaction energy dim = 2 with PBC
  '''
  neighs_sum = 0
  for i in range(q):
    neighs_sum+= lattice[(center_x+int(neighs[i,0]))%L,(center_y+int(neighs[i,1]))%L]
  return lattice[center_x,center_y]*neighs_sum

def interaction_energy_1D_PBC(q,L,lattice,neighs,center):
  '''
  Compute the interaction energy dim = 1 with PBC
  '''
  neighs_sum = 0
  for i in range(q):
    neighs_sum+= lattice[(center+int(neighs[i]))%L]
  return lattice[center]*neighs_sum

def interaction_energy_2D_positiveBC(q,L,lattice,neighs,center_x,center_y):
  '''
  Compute the interaction energy with Fixed Boundary Condition, where
  the system is sorrounded by infinite positive spin background, dim = 2
  '''
  neighs_sum = 0
  for i in range(q):
    if not is_in(center_x+int(neighs[i,0]),L)  or not is_in(center_y+int(neighs[i,1]),L):
      neighs_sum += 1
    else: 
      neighs_sum += lattice[center_x+int(neighs[i,0]),center_y+int(neighs[i,1])]
  return lattice[center_x,center_y]*neighs_sum 

def interaction_energy_1D_positiveBC(q,L,lattice,neighs,center):
  '''
  Compute the interaction energy with Fixed Boundary Condition, where
  the system is sorrounded by infinite positive spin background, dim = 1
  '''
  neighs_sum = 0
  for i in range(q):
    if not is_in(center+int(neighs[i]),L):
      neighs_sum += 1
    else: 
      neighs_sum += lattice[center+int(neighs[i])]
  return lattice[center]*neighs_sum 

def metropolis_filter_2D(lattice,energy,neighs_energy,filter_energy,center_x,center_y):
  '''
  Perform the spin flip using metropolis prescription, dim = 2
  '''
  if energy < 0:
    lattice[center_x,center_y]*=-1
  else:
    probability = random.random()
    filter = filter_energy[neighs_energy == energy]
    if filter > probability:
      lattice[center_x,center_y]*=-1

def metropolis_filter_1D(lattice,energy,neighs_energy,filter_energy,center):
  '''
  Perform the spin flip using metropolis prescription, dim = 1
  '''
  if energy < 0:
    lattice[center]*=-1
  else:
    probability = random.random()
    filter = filter_energy[neighs_energy == energy]
    if filter > probability:
      lattice[center]*=-1

def glauber_filter_2D(lattice,energy,neighs_energy,filter_energy,center_x,center_y):
  '''
  Perform the spin flip using glauber prescription, dim = 2
  '''
  probability = random.random()
  filter = filter_energy[neighs_energy == energy]
  if filter < probability:
    lattice[center_x,center_y]*=-1

def glauber_filter_1D(lattice,energy,neighs_energy,filter_energy,center):
  '''
  Perform the spin flip using glauber prescription, dim = 2
  '''
  probability = random.random()
  filter = filter_energy[neighs_energy == energy]
  if filter < probability:
    lattice[center]*=-1

def q_metropolis_2D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Metropolis prescription and PBC, dim = 2
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_2D(L)
  #sampling the starting site
  center_x, center_y = sampling_starting_spin_quadratic(L)
  #sampling q neighbours
  sampling_q_neighbours_quadratic_bounded(q,neighs,UPPER,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_2D_PBC(q,L,lattice,neighs,center_x,center_y)
  #perform the spin flip
  metropolis_filter_2D(lattice,energy,neighs_energy,filter_energy,center_x,center_y)

def q_metropolis_1D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Metropolis prescription and PBC, dim = 1
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_1D(L)
  #sampling the starting site
  center= sampling_starting_spin_linear(L)
  #sampling q neighbours
  sampling_q_neighbours_linear_bounded(q,neighs,UPPER,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_1D_PBC(q,L,lattice,neighs,center)
  #perform the spin flip
  metropolis_filter_1D(lattice,energy,neighs_energy,filter_energy,center)

def q_metropolis_infinite_2D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Metropolis prescription and PBC, with no Upper limit, dim = 2
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_2D(L)
  #sampling the starting site
  center_x, center_y = sampling_starting_spin_quadratic(L)
  #sampling q neighbours
  sampling_q_neighbours_quadratic_infinite(q,neighs,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_2D_PBC(q,L,lattice,neighs,center_x,center_y)
  #perform the spin flip
  metropolis_filter_2D(lattice,energy,neighs_energy,filter_energy,center_x,center_y)

def q_metropolis_infinite_1D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Metropolis prescription and PBC, with no Upper limit, dim = 1
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_1D(L)
  #sampling the starting site
  center= sampling_starting_spin_linear(L)
  #sampling q neighbours
  sampling_q_neighbours_linear_infinite(q,neighs,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_1D_PBC(q,L,lattice,neighs,center)
  #perform the spin flip
  metropolis_filter_1D(lattice,energy,neighs_energy,filter_energy,center)

def q_metropolis_positiveBC_2D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Metropolis prescription and Fixed Boundary Condition, where the system is 
  sorrounded by infinite positive spin background, dim = 2
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_2D(L)
  #sampling the starting site
  center_x, center_y = sampling_starting_spin_quadratic(L)
  #sampling q neighbours
  sampling_q_neighbours_quadratic_infinite(q,neighs,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_2D_positiveBC(q,L,lattice,neighs,center_x,center_y)
  #perform the spin flip
  metropolis_filter_2D(lattice,energy,neighs_energy,filter_energy,center_x,center_y)

def q_metropolis_positiveBC_1D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Metropolis prescription and Fixed Boundary Condition, where the system is 
  sorrounded by infinite positive spin background, dim = 1
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_1D(L)
  #sampling the starting site
  center= sampling_starting_spin_linear(L)
  #sampling q neighbours
  sampling_q_neighbours_linear_infinite(q,neighs,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_1D_positiveBC(q,L,lattice,neighs,center)
  #perform the spin flip
  metropolis_filter_1D(lattice,energy,neighs_energy,filter_energy,center)

def q_glauber_2D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Glauber prescription and PBC, dim = 2
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_2D(L)
  #sampling the starting site
  center_x, center_y = sampling_starting_spin_quadratic(L)
  #sampling q neighbours
  sampling_q_neighbours_quadratic_bounded(q,neighs,UPPER,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_2D_PBC(q,L,lattice,neighs,center_x,center_y)
  #perform the spin flip
  glauber_filter_2D(lattice,energy,neighs_energy,filter_energy,center_x,center_y)

def q_glauber_1D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Glauber prescription and PBC, dim = 1
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_1D(L)
  #sampling the starting site
  center= sampling_starting_spin_linear(L)
  #sampling q neighbours
  sampling_q_neighbours_linear_bounded(q,neighs,UPPER,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_1D_PBC(q,L,lattice,neighs,center)
  #perform the spin flip
  glauber_filter_1D(lattice,energy,neighs_energy,filter_energy,center)

def q_glauber_infinite_2D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Glauber prescription and PBC, with no Upper limit, dim = 2
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_2D(L)
  #sampling the starting site
  center_x, center_y = sampling_starting_spin_quadratic(L)
  #sampling q neighbours
  sampling_q_neighbours_quadratic_infinite(q,neighs,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_2D_PBC(q,L,lattice,neighs,center_x,center_y)
  #perform the spin flip
  glauber_filter_2D(lattice,energy,neighs_energy,filter_energy,center_x,center_y)

def q_glauber_infinite_1D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Glauber prescription and PBC, with no Upper limit, dim = 1
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_1D(L)
  #sampling the starting site
  center= sampling_starting_spin_linear(L)
  #sampling q neighbours
  sampling_q_neighbours_linear_infinite(q,neighs,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_1D_PBC(q,L,lattice,neighs,center)
  #perform the spin flip
  glauber_filter_1D(lattice,energy,neighs_energy,filter_energy,center)

def q_glauber_positiveBC_2D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Glauber prescription and Fixed Boundary Condition, where the system is 
  sorrounded by infinite positive spin background, dim = 2
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_2D(L)
  #sampling the starting site
  center_x, center_y = sampling_starting_spin_quadratic(L)
  #sampling q neighbours
  sampling_q_neighbours_quadratic_infinite(q,neighs,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_2D_positiveBC(q,L,lattice,neighs,center_x,center_y)
  #perform the spin flip
  glauber_filter_2D(lattice,energy,neighs_energy,filter_energy,center_x,center_y)

def q_glauber_positiveBC_1D(lattice,q,alpha,neighs,filter_energy,neighs_energy):
  '''
  Perform a single spin flip of the Dynamical Levy Lattice Algorihtm, using
  Glauber prescription and Fixed Boundary Condition, where the system is 
  sorrounded by infinite positive spin background, dim = 1
  ''' 
  L = len(lattice)
  #define limits
  UPPER, LOWER = define_limits_1D(L)
  #sampling the starting site
  center= sampling_starting_spin_linear(L)
  #sampling q neighbours
  sampling_q_neighbours_linear_infinite(q,neighs,LOWER,alpha)
  #compute interaction energy
  energy = interaction_energy_1D_positiveBC(q,L,lattice,neighs,center)
  #perform the spin flip
  glauber_filter_1D(lattice,energy,neighs_energy,filter_energy,center)

def create_energy(q,spin,Prescription):
  '''
  Create possible neighbours' energies array, where 'q' is the number of neighbours and 'spin' stores the possible values of spins;
  for example in the Ising model: spin = [-1,1]
  '''
  conf = (itertools.combinations_with_replacement(spin,q))
  energy = []
  for i in list(conf):
    energy.append(sum(i))

  if Prescription == 'metropolis':
    return energy[int(len(energy)/2):]
  else:
    return energy

def make_Temperature_file(Ls,path,Ts,steps):
  '''
  Create a file "path/Ts.npy" which stores the temperatures
  and a file "path/Ts.txt" which could be useful for future simulations.
  '''
  temper = np.empty((len(Ls),len(Ts[Ls[0]])))
  L_idx = 0
  for L1 in Ls:
      temper[L_idx] = Ts[L1]
      L_idx+=1
  np.save(path+'Ts.npy',temper)
  with open(path+'Ts.txt', 'w') as f:
      L_idx = 0
      print('number of Temperature, num_T = ',len(Ts[Ls[0]]),file = f)
      print('Ts = {',file = f)
      for L1 in Ls:
          print(L1,': np.linspace(',temper[L_idx][0],', ',temper[L_idx][-1],',num_T),',file = f)
          L_idx+=1
      print('}',file = f)
    
def choose_dynamics(Parameters):
    '''
    Choose the requested dynamics
    '''
  if Parameters['Prescription'] == 'metropolis':
    if Parameters['Boundary Condition'] == 'PBC':
      if Parameters['Dimension'] == 1:
        dynamics = q_metropolis_1D
      elif Parameters['Dimension'] == 2:
        dynamics = q_metropolis_2D
    elif Parameters['Boundary Condition'] == 'infinite PBC':
      if Parameters['Dimension'] == 1:
        dynamics = q_metropolis_infinite_1D
      elif Parameters['Dimension'] == 2:
        dynamics = q_metropolis_infinite_2D
    elif Parameters['Boundary Condition'] == 'positiveBC':
      if Parameters['Dimension'] == 1:
        dynamics = q_metropolis_positiveBC_1D
      elif Parameters['Dimension'] == 2:
        dynamics = q_metropolis_positiveBC_2D
    else:
      print('Erros: not existing boundary condition!')
  elif Parameters['Prescription'] == 'glauber':
    if Parameters['Boundary Condition'] == 'PBC':
      if Parameters['Dimension'] == 1:
        dynamics = q_glauber_1D
      elif Parameters['Dimension'] == 2:
        dynamics = q_glauber_2D
    elif Parameters['Boundary Condition'] == 'infinite PBC':
      if Parameters['Dimension'] == 1:
        dynamics = q_glauber_infinite_1D
      elif Parameters['Dimension'] == 2:
        dynamics = q_glauber_infinite_2D
    elif Parameters['Boundary Condition'] == 'positiveBC':
      if Parameters['Dimension'] == 1:
        dynamics = q_glauber_positiveBC_1D
      elif Parameters['Dimension'] == 2:
        dynamics = q_glauber_positiveBC_2D
    else:
      print('Erros: not existing boundary condition!')
  else:
    print('Error: not existing dynamics!')
  print('Chosen dynamics: ', dynamics)
  return dynamics
