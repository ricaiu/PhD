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


np.random.seed(23)



#utilities


univ_bis = True

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
dropout = True


callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1,min_delta=0.001)
opt = optimizers.Adam(learning_rate=lr)












input_shape=(L,L,1)



T = []
for i in range(Nconf):
    T.append([0,1])
for i in range(Nconf):
    T.append([1,0])


y_label = np.array(T)


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
#x2=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/univ/SW_LATT_critic_2d_10to3.dat' ,dtype='int32')

x2=np.reshape(x2,(y,L,L))
print(x1.shape)
print(x2.shape)


m1=[]
m2=[]

for i in range(y):
    m1.append(x1[i].sum()/L**2)
    m2.append(x2[i].sum()/L**2)


m1 = np.array(m1)
m2 = np.array(m2)
bins = 200
plt.figure(6)
plt.hist(m1,color = 'blue',alpha=0.4, bins = bins,label = '2D')
plt.hist(m2,color = 'yellow',alpha=0.6, bins = bins, label = '3D')
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

for i in range(len(x1)):
    m = x1[i].sum()/(L**2)
    if m > 0:
        x1[i] = - x1[i]
    m = x2[i].sum()/(L**2)
    if m > 0:
        x2[i] = - x2[i]




X = np.concatenate((x1,x2))




print('ylabel.shape ',y_label.shape)
print('X.shape ',X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y_label,test_size=0.2,random_state=302, shuffle=True)
X_train = tf.expand_dims(X_train, axis=-1)

X_test = tf.expand_dims(X_test, axis=-1)
print('shape of X train: ', X_train.shape)
print('shape of y train: ', y_train.shape)
print('shape of X test: ', X_test.shape)
print('shape of y test: ', y_test.shape)


network = models.Sequential() #initialize the neural network




network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride,
                                        activation='relu', input_shape=input_shape))
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
#Train the network

network.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

history = network.fit(X_train, y_train, epochs=10000, batch_size=32, callbacks = callbacks,validation_data=(X_test, y_test))# validation_split=0.15)#

train_weights = []
train_bias = []


layer = []
for i in range(Channel):
    layer.append(network.layers[i])

for i in range(Ndense):
    layer.append(network.layers[-Ndense+i])


for i in layer:
    train_weights.append(np.array(i.get_weights()[0]))
    train_bias.append(np.array(i.get_weights()[1]))

train_weights = np.array(train_weights)
train_bias = np.array(train_bias)


# for i in range(len(train_weights)):
#     print(train_weights[i].shape)
#     print(train_bias[i].shape)

for j in range(Channel):
    filters = train_weights[j]

    filters_bias = train_bias[j]
    f1 = open(directory+"filter_matrix_univ_test_"+str(j)+".txt", "w")


    for i in range(Channel):
        # get the filter
        f = filters[:, :, :, i]
        np.savetxt(f1,f[:, :, 0])


    f1.close()
    f = open(directory+"filter_bias_univ_test_"+str(j)+".txt", "w")
    np.savetxt(f,filters_bias)
    f.close()


for i in np.arange(Channel,len(train_weights),1):
    weights = train_weights[i]
    bias = train_bias[i]

    f = open(directory+"weight_matrix_univ_test_"+str(i)+".txt", "w")
    np.savetxt(f,weights)
    f.close()

    f = open(directory+"bias_univ_test_"+str(i)+".txt", "w")
    np.savetxt(f,bias)
    f.close()



# if (univ_bis):
#     f = open(directory+"X_test.txt", "w")
#     np.savetxt(f,X_test)
#     f.close()
#     f = open(directory+"y_test.txt", "w")
#     np.savetxt(f,y_test)
#     f.close()
#
# plt.figure(4)
# plt.imshow(train_weights, cmap ='Greys' )
# plt.colorbar()
# plt.figure(7)
# plt.imshow(last_neuron, cmap ='Greys' )
# plt.colorbar()
#

# Plot loss (y axis) and epochs (x axis) for training set and validation set
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.epoch,np.array(history.history['loss']),label='Train loss')
plt.plot(history.epoch,np.array(history.history['val_loss']),label = 'Val loss')
plt.legend()

# Plot loss (y axis) and epochs (x axis) for training set and validation set
plt.figure(2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.epoch,np.array(history.history['loss']),label='Train loss')
plt.plot(history.epoch,np.array(history.history['val_loss']),label = 'Val loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()

# Plot loss (y axis) and epochs (x axis) for training set and validation set
plt.figure(3)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.epoch,np.array(history.history['accuracy']),label='Train acc')
plt.plot(history.epoch,np.array(history.history['val_accuracy']),label = 'Val acc')
plt.legend()

plt.show()
