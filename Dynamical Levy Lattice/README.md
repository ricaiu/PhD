---
Dynamical Levy Lattice
---

In this repository you can find the code used for creating the data, which were used in the paper "Critical dynamics of long range models on Dynamical Lévy Lattices", currently submitted on PRB (APS). You can find a preprint version here: https://arxiv.org/abs/2303.18057..\
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
--> C++\
   In this folder you can find the same algorithm but in C++, both for 1D and 2D case. Here I implemented also a version with OMP, which allows you to run the code in parallel CPUs, in order to speed up simulations. On our machines, the C++ code seems faster than the Python one by a O(50) times in computational time, so I recommend you to use it! (But the Python code is more readable!) In this folder there is also a famous pseudorandom number generator the SFMT.\
   \
 Please, feel free to use this code for you research or curiosity! \
 Contact me for more details or if you find some bugs!
