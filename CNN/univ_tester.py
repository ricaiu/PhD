from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from keras.regularizers import l2

def canonical(size, index):
    arr = np.zeros(size)
    arr[index] = 1.0
    return arr

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


np.random.seed(23)



univ_bis = False

magne = '02'


if (univ_bis):
    directory = 'univ_test_'+magne+'/'
else:
    directory = 'univ_test/'


L= 32

Nconf = 3*10**4


Ndense = 2

Nf=4
Channel=1
Stride=2#int(L/4)
l2_r=0.01
lr=0.00001


dropout_const = 0.3
dropout = False


callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1,min_delta=0.001)
opt = optimizers.Adam(learning_rate=lr)












input_shape=(L,L,1)


if (univ_bis):
    x1=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_2d_'+magne+'.dat' ,dtype='int32')
else:
    x1=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_2d.dat' ,dtype='int32')
y=int(len(x1)/L**2)
x1=np.reshape(x1,(y,L,L))



if (univ_bis):
    x2=np.loadtxt('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_3d_'+magne+'.txt')
else:
    x2=np.loadtxt('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_3d.txt')


x2=np.reshape(x2,(y,L,L))
print(x1.shape)
print(x2.shape)


if (univ_bis):
    x1_test=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_2d_'+magne+'_test.dat' ,dtype='int32')
else:
    x1_test=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_2d_test.dat' ,dtype='int32')
y=int(len(x1_test)/L**2)
x1_test=np.reshape(x1_test,(y,L,L))



if (univ_bis):
    x2_test=np.loadtxt('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_3d_'+magne+'_test.txt')
else:
    x2_test=np.loadtxt('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_3d_test.txt')


x2_test=np.reshape(x2_test,(y,L,L))
print(x1_test.shape)
print(x2_test.shape)

m1_test=[]
m2_test=[]

for i in range(y):
    m1_test.append(x1_test[i].sum()/L**2)
    m2_test.append(x2_test[i].sum()/L**2)



m1_test = np.array(m1_test)
m2_test = np.array(m2_test)
bins = 200
plt.figure(1)
plt.hist(m1_test,color = 'blue',alpha=0.8, bins = bins,label = '2D',density = True)
plt.hist(m2_test,color = 'red',alpha=0.8, bins = bins, label = '3D',density = True)
plt.legend(loc = 'best')




# m1_test = m1[(m1>-0.2)*(m1<0.2)]
# m2_test = m2[(m2>-0.2)*(m2<0.2)]







# plt.show()
# plt.figure(7)
# plt.plot(m1-m2,color = 'blue')
# for j in np.arange(8,17,2):
#     plt.figure(j)
#     plt.imshow(x1[100+100*j])
#     plt.figure(j+1)
#     plt.imshow(x2[100+100*j])
# plt.show()

for i in range(len(x1_test)):
    m = x1_test[i].sum()/(L**2)
    if m > 0:
        x1_test[i] = - x1_test[i]
    m = x2_test[i].sum()/(L**2)
    if m > 0:
        x2_test[i] = - x2_test[i]




x1 = tf.expand_dims(x1, axis=-1)
x2 = tf.expand_dims(x2, axis=-1)

x1_test = tf.expand_dims(x1_test, axis=-1)
x2_test = tf.expand_dims(x2_test, axis=-1)

T = []
for i in range(Nconf):
    T.append([0,1])#2d
for i in range(Nconf):
    T.append([1,0])#3d


y_label = np.array(T)



network = models.Sequential() #initialize the neural network


network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride, activation='relu', input_shape=input_shape))
#network.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride, activation='relu', padding='valid'))



#dropout: spegne casualmente qualche neurone
indexlayer = 0
if dropout:
    network.add(layers.Dropout(dropout_const))
    indexlayer += 1
network.add(layers.Flatten())



network.add(layers.Dense(64, activation='relu'))

# network.add(layers.Dense(16, activation='relu'))

network.add(layers.Dense(2, activation='sigmoid'))



network.summary() # Shows information about layers and parameters of the entire network


print(len(network.layers))

layer = []
for i in range(Channel):
    layer.append(network.layers[i])

for i in range(Ndense):
    layer.append(network.layers[-Ndense+i])
#layer = [network.layers[0],network.layers[-2],network.layers[-1]]
print(layer)


for i in range(Channel):

    filter = np.loadtxt('/home/riccardo/DOTTORATO/CODE/CNN/'+directory+'filter_matrix_univ_test_'+str(i)+'.txt')
    filter_bias = np.loadtxt('/home/riccardo/DOTTORATO/CODE/CNN/'+directory+'filter_bias_univ_test_'+str(i)+'.txt')
    filter = tf.expand_dims(filter, axis=-1)
    filter = tf.expand_dims(filter, axis=-1)
    filter_bias = tf.expand_dims(filter_bias, axis=-1)



    layer[i].set_weights([filter,filter_bias])



for i in np.arange(Channel,len(layer),1):

    weight = np.loadtxt('/home/riccardo/DOTTORATO/CODE/CNN/'+directory+'weight_matrix_univ_test_'+str(i)+'.txt')
    bias = np.loadtxt('/home/riccardo/DOTTORATO/CODE/CNN/'+directory+'bias_univ_test_'+str(i)+'.txt')


    layer[i].set_weights([weight,bias])






y = network.predict( x2_test, verbose = 1)
print(y)
plt.figure(2)
plt.imshow(y[:10])
plt.colorbar()

tresh = [0.5,0.6,0.7]#,0.8,0.9]

bin_err = 20

for i in tresh:

    index = y[:,0]<= i
    print('Number of error for acc < ',i, ' : ',x1_test[index].shape[0])
    print('Percentage of error for acc < ',i, ' : ',100*x1_test[index].shape[0]/x1_test.shape[0],'%')
    plt.figure(1)
    plt.hist(m1_test[index],alpha=0.5, bins = bin_err,label = '2D acc <'+str(i),density = True)
    plt.legend(loc = 'best')


y = network.predict( x1_test, verbose = 1)
print(y)

plt.figure(3)
plt.imshow(y[:10])
plt.colorbar()





for i in tresh:

    index = y[:,1]<= i
    print('Number of error for acc < ',i, ' : ',x2_test[index].shape[0])
    print('Percentage of error for acc < ',i, ' : ',x2_test[index].shape[0]/x2_test.shape[0],'%')
    plt.figure(1)
    plt.hist(m2_test[index],alpha=0.5, bins = bin_err,label = '3D acc <'+str(i),density = True)
    plt.legend(loc = 'best')


# y = network.predict( x1, verbose = 1)
# print(y)
#
# plt.figure(4)
# plt.imshow(y[:10])
# plt.colorbar()
#
#
# y = network.predict( x2, verbose = 1)
# print(y)
#
# plt.figure(5)
# plt.imshow(y[:10])
# plt.colorbar()

plt.show()
