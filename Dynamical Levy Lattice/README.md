---
Dynamical Levy Lattice
---
---
1) Choose a random spin $S_i$  
2) Draw $q$ neighbours $S_{r_j}$, each at distance $r_j$ from $S_i$, using as probability distribution $P(r)\sim r^{-(d+\sigma)}$  
3) Calculate the energy difference $\Delta E = 2\cdot J \cdot S_i \sum_{j}  S_{r_j}$  
4) Flip $S_i$ following the wanted prescription (Glauber or Metropolis)  

--- 

In this repository you can find the code used for creating the data, which were used  in the paper ( Paper is not finished yet).\
In a nutshell we provide an algorithm which is able to generate configurations of a Long Range Ising Model, where the interaction decays with the distance following a power-law $r^{-(d+\sigma)}$. \
The repository is organized as follow:\
--> functions.py\
  Here you find all the functions that will be used in the main program.\
--> parameters.py\
   Here you can personalize the simulation with the wanted parameters (as temperature, dimension, sizes,...) and choose which prescription for the dynamics you want.\
--> main.py\
   This is the main program, the only one you have to run, after changing the parameters.\
--> main.ipynb\
   This is a Colab Notebook where all three programs mentioned before are present. Here you can see all the algorithm details in a more readability way.\
   \
 Please, feel free to use this code for you research or curiosity! \
 Contact me for more details or if you find some bugs!\
 If you will use this for Academic purposes I will appreciate a tiny and simple citation :)
