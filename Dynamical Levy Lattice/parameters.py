#For each size choose the temperature range and save it in this dictionary:
Ts  = {8 : np.linspace(1,2,10)}


Parameters = {
    'Prescription' : 'metropolis', #'metropolis' 'glauber'
    'Boundary Condition' : 'PBC', #'PBC', 'infinite_PBC', 'positiveBC
    'Start' : 'hot',
    'Dimension' : 1,
    'Sigma' : 1,
    'Neighbours' : 4,
    'Sizes' : np.array([8,]),
    'Temperatures' : Ts ,
    'Autocorrelation' : 100,
    'Steps' : 100,
    'Path' : '',
    'File Name' : '',
    'Output file name' :  'test'

}
