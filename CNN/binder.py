import numpy as np
import matplotlib.pyplot as plt



def binder(sample, L):
	m2=0
	m4=0
	# m2 = []
	# m4 = []
	for i in range(len(sample)):
		m2+=(sample[i].sum()/L**2)**2
		m4+=(sample[i].sum()/L**2)**4
		# m2.append((sample[i].sum()/L**2)**2)
		# m4.append((sample[i].sum()/L**2)**4)

	m2 = m2/len(sample)
	m4 = m4/len(sample)
	return 0.5*(3-m4/(m2**2))



def magn(sample, l):
	magne = []
	for i in range(len(sample)):
		m = sample[i].sum()/l**2
		if m >0:
			magne.append(m)
		else:
			magne.append(-m)
	magne = np.array(magne)
	return magne.mean()
	# m2 = np.array(m2)
	# m4 = np.array(m4)
	# m2_mean = m2.mean()
	# m4_mean = m4.mean()
	# m2_dev = m2.std()
	# m4_dev = m2.std()
	# bind = 0.5*(3-m4_mean/(m2_mean**2))
	# bind_dev = 0.5*(m4_dev/m4_mean+2*m2_mean*m2_dev/m2_mean**2)*m4_mean/(m2_mean**2)
	# return bind, bind_dev


temp_sample = 1000

temp_2d = np.arange(2.26,2.356,0.005)

temp_3d = np.arange(4.65,4.746,0.005)





L=[16,32,64,128]


for l in L:
	X_2d=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_binder_2d_'+str(l)+'.dat' ,dtype='int32')



	y=int(len(X_2d)/l**2)
	X_2d=np.reshape(X_2d,(y,l**2))
	X_2d=np.reshape(X_2d,(int(y/temp_sample),temp_sample,l**2))


	print(X_2d.shape)

	binder_2d = []
	magn_2d = []

	for i in range(len(X_2d)):
		binder_2d.append(binder(X_2d[i],l))
		magn_2d.append(magn(X_2d[i],l))

	plt.figure(1)
	plt.plot(temp_2d,binder_2d,linestyle='-', marker='.',label=str(l))
	plt.figure(2)
	plt.plot(temp_2d,magn_2d,linestyle='-', marker='.',label=str(l))

plt.figure(1)
plt.title('Binder cumulant 2D, Ising model')
plt.xlabel('T')
plt.ylabel("Binder's cumulant")
plt.legend( loc = 'best')
plt.axvline(x=2.325)
plt.figure(2)
plt.title('|magnetization| 2D, Ising model')
plt.xlabel('T')
plt.ylabel("|m|")
plt.legend( loc = 'best')
plt.axvline(x=2.325)


L=[16,32,64]


for l in L:
	X_2d=np.loadtxt('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_binder_3d_'+str(l)+'.txt' )


	X_2d=np.reshape(X_2d,(len(temp_3d),temp_sample,l**2))
	print(X_2d.shape)

	binder_2d = []
	magn_2d = []

	for i in range(len(X_2d)):
		binder_2d.append(binder(X_2d[i],l))
		magn_2d.append(magn(X_2d[i],l))

	plt.figure(3)
	plt.plot(temp_3d,binder_2d,linestyle='-', marker='.',label=str(l))
	plt.figure(4)
	plt.plot(temp_3d,magn_2d,linestyle='-', marker='.',label=str(l))

plt.figure(3)
plt.title('Binder cumulant 3D slice, Ising model')
plt.xlabel('T')
plt.ylabel("Binder's cumulant")
plt.legend( loc = 'best')
plt.axvline(x=4.715)
plt.figure(4)
plt.title('|magnetization| 3D, Ising model')
plt.xlabel('T')
plt.ylabel("|m|")
plt.legend( loc = 'best')
plt.axvline(x=4.715)


plt.show()
