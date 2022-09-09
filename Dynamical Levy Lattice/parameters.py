#For each size choose the temperature range and save it in this dictionary:
Ts  = {8 : np.linspace(1,2,10),
       16: np.linspace(1.2,1.8,10)}


Parameters = {
    'Prescription' : 'metropolis', #The actual possibilities are: 'metropolis' 'glauber'
    'Boundary Condition' : 'PBC', #The actual possibilities are: 'PBC', 'infinite PBC', 'positiveBC'
    'Start' : 'hot', #The actual possibilities are: 'hot', 'cold'
    'Dimension' : 1,
    'Sigma' : 1,
    'Neighbours' : 4,
    'Sizes' : np.array([8,]),
    'Temperatures' : Ts ,
    'Autocorrelation' : 100,
    'Steps' : 100,
    'Path' : 'saving_path/',
    'File Name' : 'filename',
    'Output file name' :  'output_filename'

}
