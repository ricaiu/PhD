Implementing q_random Ising model.

Contents:

- q_random_ising.py \
    Simulate a q random Ising model via glauber or metropolis sampling. At each simulation step, one site is randomly chosen with uniform distribution and its neighbours
    with power-law distribution, governed by a parameter σ.\
    It generates "m_exp#.npy" file, with # the indicative number of the test and contains the magnetization of the lattice. It's an array of shape (len(Temperature), len(Sample)).

- variables.py \
    Contains reference to the data you want to analyze, variables and utilities.
 
- bootstrap.py\
    It takes the files "m_exp#.npy" and calculates the mean and the standard deviation of: Magnetization (m.npy,dm.npy), Supscetivity (chi.npy,dchi.npy) and Binder's cumulant (binder.npy, dbinder.npy). For susceptivity and Binder the standard deviation is calculated via bootstrap method.

- simple_plot.py\
    It takes the data generated via "bootstrap.py" and simply plot M, Chi and Binder vs T.
 
- tc_binder_ext.py\
    It takes "binder.npy" and "Tc_guess" from "variable.py". First it linearizes the nearest Binder points around "Tc_guess" (the number of points taken can be chosen) with linear regression; then it chooses the temperature crossing point between Binder at L and 2L, Tc(L), via the parameter "threesh". Finally it does a linear fitting log(Tc(L)) = a*log(1/L)+b.\
The value of b is the extraction of the critical temperature Tc and it is saved in the file "Tc.npy"

- eta_nu_extraction.py\
 It takes "chi.npy" and "Tc.npy" generated via "tc_binder_ext.py". Similar to "tc_binder_ext.py", it takes the nearest Chi points around the maximum of Chi and performs a regression with a parable. Then it takes, for each lenghts L, as maximum of Chi (chi_max(L)) the maximum value of the fitted parable and as T_max(L) the value of the temperature for this point.\
 Finally it performs two linear regression:\
    log(T_max(L)-Tc) = a*log(L)+b. The value of "a" is -1/ʋ \
   log(chi_max(L)) = a*log(L) +b. The value of "a" is 2-η\
 These two critical exponents (ʋ and 2-η) are saved in the files "nu.npy" and "2_min_eta.npy"

- scaling_plot.py\
It takes the critical exponents and plot the Magnetization, Susceptivity adn Binder's cumulant with the expected scaling.

