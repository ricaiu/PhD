import numpy as np

L=3

nnr = np.zeros(L**3, dtype = int)
nnl = np.zeros(L**3, dtype = int)
nnu = np.zeros(L**3, dtype = int)
nnd = np.zeros(L**3, dtype = int)
nnf = np.zeros(L**3, dtype = int)
nnb = np.zeros(L**3, dtype = int)




for index in range(L):



    for MYi in np.arange(index*L**2,index*L**2+L**2-L,1):
        nnr[MYi] = int(MYi+L)

    for MYi in np.arange(index*L**2,index*L**2+L,1):
        nnr[L**2-L+MYi] = int(MYi)

    for MYi in np.arange(index*L**2+L,index*L**2+L**2,1):
        nnl[MYi] = int(MYi-L)


    for MYi in np.arange(index*L**2,index*L**2+L,1):
        nnl[MYi] = int(L**2-L+ MYi)

    for MYi in np.arange(index*L**2,index*L**2+L**2,1):
        nnu[MYi] = int(MYi-1)

    for MYi in np.arange(index*L,index*L+L,1):
        nnu[MYi*L] = int((MYi+1)*L-1)

    for MYi in np.arange(index*L**2,index*L**2+L**2,1):
        nnd[MYi] = int(MYi+1)

    for MYi in np.arange(index*L,index*L+L,1):
        nnd[(MYi+1)*L-1] = int(MYi*L)



for MYi in range(L**3-L**2):

    nnf[MYi] = MYi+L**2

for MYi in np.arange(L**3-L**2,L**3,1):

    nnf[MYi] = MYi - L**3+L**2

for MYi in range(L**2):

    nnb[MYi] = MYi+L**3-L**2

for MYi in np.arange(L**2,L**3,1):

    nnb[MYi] = MYi-L**2



for i in range(len(nnr)):
    print('nnr(',i,') = ',nnr[i])
for i in range(len(nnr)):
    print('nnl(',i,') = ',nnl[i])
for i in range(len(nnr)):
    print('nnu(',i,') = ',nnu[i])
for i in range(len(nnr)):
    print('nnd(',i,') = ',nnd[i])
for i in range(len(nnr)):
    print('nnf(',i,') = ',nnf[i])
for i in range(len(nnr)):
    print('nnb(',i,') = ',nnb[i])
