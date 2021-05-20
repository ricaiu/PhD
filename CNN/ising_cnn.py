from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from keras.regularizers import L2

def canonical(size, index):
    arr = np.zeros(size)
    arr[index] = 1.0
    return arr

def hot_encoding(y, N, beta,dn,bmin):
    for b in beta:
        if (b<0):
            hot_cod = canonical(N,0)
        elif(b>=1):
            hot_cod = canonical(N,N-1)
        else:
            for n in range(N):
                if ((b>= bmin+n*dn) and (b< bmin +(n+1)*dn)):
                    hot_cod=canonical(N,n)
                    #print(n*dn)
        y.append(hot_cod)


# def hot_encoding(y, N, beta,dn,bmin):
#     for b in beta:
#         if (b<0):
#             hot_cod = canonical(N,0)
#         elif(b>=1):
#             hot_cod = canonical(N,N-1)
#         else:
#             for n in range(N):
#                 if ((b>=n*dn) and (b<(n+1)*dn)):
#                     hot_cod=canonical(N,n+1)
#                     #print(n*dn)
#         y.append(hot_cod)


def hot_encodingT(y, N,T, Tmin, Tmax,dn):
    for b in T:
        #print(b)
        for n in range(N):
            if ((b>=Tmin +n*dn) and (b<Tmin +(n+1)*dn)):
                hot_cod=canonical(N,n)
                #print(Tmin +n*dn)
        y.append(hot_cod)







#utilities

nsave=1
L= 16
Tstart = 0.1
Tend = 5.0
Nconf = 10**4
delta = (Tend - Tstart)/Nconf
N = 100
dnT=(Tend - Tstart)/N#1/(N-2)#mettere delta
dn=1/(N-2)
Nf=3
Channel=5
Stride=4#int(L/4)
l2_r=0.0001


lr=0.00001

dropout_const = 0.3
dropout = True

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1,min_delta=0.005)

T = []

T_fromfile = np.genfromtxt('/home/riccardo/DOTTORATO/CODE/SW/provaT_cnn.txt' )
for i in np.arange(0,len(T_fromfile),3):
    for j in range(nsave):
        T.append(T_fromfile[i])

T = np.array(T)

beta = np.array([1/i for i in T])


X=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/SW_LATT_cnn.dat' ,dtype='int32')
X=np.array(X)

y=int(len(X)/L**2)
X=np.reshape(X,(y,L,L))

for i in range(len(X)):
    m = X[i].sum()/(L**2)
    if m > 0:
        X[i] = - X[i]

#norm
#X = 0.5*(X+1)


y_label= []



#hot_encoding(y_label,N,beta,dn,beta[-1])
hot_encodingT(y_label,N,T,Tstart,Tend,dnT)
y_label = np.array(y_label)
y_label = np.reshape(y_label,(len(T),N))

print('ylabel.shape ',y_label.shape)
print('X.shape ',X.shape)
#
#

#
#
input_shape=(L,L,1)

network = models.Sequential() #initialize the neural network
'''
network.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, padding="same"))
network.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
network.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
network.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
network.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
network.add(layers.Flatten())
network.add(layers.Dense(84, activation='relu'))
'''

network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride,
                activation='relu', input_shape=input_shape,
                kernel_regularizer = L2(l2_r)))


#dropout: spegne casualmente qualche neurone
indexlayer = 0
if dropout:
    network.add(layers.Dropout(dropout_const))
    indexlayer += 1
network.add(layers.Flatten())
indexlayer += 1



# network.add(layers.Dense(N, activation='relu'))


network.add(layers.Dense(N, activation='softmax',kernel_regularizer = L2(l2_r)))



network.summary() # Shows information about layers and parameters of the entire network


print(len(network.layers))
print(network.layers)


X_train, X_test, y_train, y_test = train_test_split(X, y_label,test_size=0.2,
                                                    random_state=302, shuffle=True)
X_train = tf.expand_dims(X_train, axis=-1)
#X_test = tf.expand_dims(X_test, axis=-1)
print('shape of X train: ', X_train.shape)
print('shape of y train: ', y_train.shape)
print('shape of X test: ', X_test.shape)
print('shape of y test: ', y_test.shape)



#Train the network
opt = optimizers.Adam(learning_rate=lr)
network.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

history = network.fit(X_train, y_train, epochs=10000, batch_size=8,validation_split=0.1,
                        callbacks = callbacks)# validation_data=(X_test, y_test))#

#print(opt.iterations)
train_weights, dense_biases=np.array(network.layers[3].get_weights())
filters, biases = network.layers[0].get_weights()

print('FILTER_SHAPE: ',filters.shape)
print('biases_shape: ',biases.shape)
print('dense_biases: ', dense_biases.shape)



f2 = open('cnn_data/x_test.txt' , 'w')

for i in X_test:
    np.savetxt(f2,i)

f2.close()
f2 = open('cnn_data/y_test.txt' , 'w')

np.savetxt(f2,y_test)

f2.close()

#3315
f1 = open("cnn_data/filter_matrix.txt", "w")
# plot first few filters
n_filters, ix = Channel, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    # plot each channel separately

    # specify subplot and turn of axis
    ax = plt.subplot(n_filters, 3, ix)
    ax.set_xticks([])
    ax.set_yticks([])
    # plot filter channel in grayscale
    plt.imshow(f[:, :, 0],cmap ='coolwarm' )
    np.savetxt(f1,f[:, :, 0])
    plt.colorbar()
    ix += 1
f1.close()

f = open("cnn_data/weight_matrix.txt", "w")
np.savetxt(f,train_weights)
f.close()

f = open('cnn_data/dense_biases.txt', 'w')
np.savetxt(f, dense_biases)
f.close()
f = open('cnn_data/biases.txt', 'w')
np.savetxt(f, biases)
f.close()

plt.figure(4)
plt.imshow(train_weights, cmap ='Greys' )
plt.colorbar()

# plt.figure(3)
# plt.imshow(filter[0])
# plt.colorbar()

#print('accuracy: ',history.history['accuracy'] )
#plot
plt.figure(5)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper left')
plt.figure(7)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xscale('log')
plt.yscale('log')
plt.legend(['Training Set', 'Validation Set'], loc='upper left')


plt.figure(6)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.ylabel("testing accuray")
plt.xlabel("epoch")
plt.legend(['Training Set', 'Validation Set'], loc='upper left')

plt.show()
X_test = tf.expand_dims(X_test, axis=-1)
y_predict = network.predict( X_test, verbose = 1)

f = open('cnn_data/y_predict.txt', 'w')
np.savetxt(f,y_predict)
f.close()
