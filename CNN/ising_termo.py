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


#choose 'High' or 'Low' T
mode = 'Low'
#utilities


L= 32
Tstart = 0.1
Tend = 0.2
Nconf = 10**4


Nf=2
Channel=6
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

if (mode == 'High'):

    Tmin = 2.4
    temperature = np.arange(Tmin,5.1,0.1)

    X=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/termo/SW_LATT_termo_highT.dat' ,dtype='int32')
    y=int(len(X)/L**2)
    X=np.reshape(X,(y,L,L))

    for i in range(len(X)):
        m = X[i].sum()/(L**2)
        if m > 0:
            X[i] = - X[i]

    initial_temps = [2.4,2.8,3.2,3.6,4.0,4.4,5.0]

    for initial_temp in initial_temps :

        acc = []
        val_acc = []

        initial_index = int((10*(initial_temp-Tmin))*Nconf)

        x1 = X[ initial_index : initial_index + Nconf ]

        for t in temperature:
            index = int(10*(t-Tmin))*Nconf
            x2 = X[ index : index + Nconf ]
            Xconf = np.concatenate((x1,x2))


            print('ylabel.shape ',y_label.shape)
            print('X.shape ',Xconf.shape)

            X_train, X_test, y_train, y_test = train_test_split(Xconf, y_label,test_size=0.15,random_state=302, shuffle=True)
            X_train = tf.expand_dims(X_train, axis=-1)
            X_test = tf.expand_dims(X_test, axis=-1)
            print('shape of X train: ', X_train.shape)
            print('shape of y train: ', y_train.shape)
            print('shape of X test: ', X_test.shape)
            print('shape of y test: ', y_test.shape)


            network = models.Sequential() #initialize the neural network


            network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride, activation='relu', input_shape=input_shape))
            #network.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
            network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride, activation='relu', padding='valid'))



            #dropout: spegne casualmente qualche neurone
            indexlayer = 0
            if dropout:
                network.add(layers.Dropout(dropout_const))
                indexlayer += 1
            network.add(layers.Flatten())



            network.add(layers.Dense(64, activation='relu'))

            network.add(layers.Dense(16, activation='relu'))

            network.add(layers.Dense(2, activation='sigmoid'))



            network.summary() # Shows information about layers and parameters of the entire network


            print(len(network.layers))
            #Train the network

            network.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

            history = network.fit(X_train, y_train, epochs=10000, batch_size=4, validation_data=(X_test, y_test),callbacks = callbacks)# validation_split=0.1)#

            acc.append(history.history["accuracy"][-1])
            val_acc.append(history.history["val_accuracy"][-1])
            print(acc)
            print(val_acc)
            print('end of T = ', t)


        # plt.figure(1)
        # plt.plot(acc)
        # plt.plot(val_acc)

        f = open('termo_data/acc_termo_'+str(int(10*initial_temp))+'.txt', "w")
        np.savetxt(f,acc)
        f.close()
        f = open('termo_data/val_acc_termo_'+str(int(10*initial_temp))+'.txt', "w")
        np.savetxt(f,val_acc)
        f.close()


if (mode == 'Low'):

    Tmin = 0.1
    Tmax = 2.3
    temperature = np.arange(Tmin,2.4,0.1)
    X=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/termo/SW_LATT_termo_lowT_'+str(L)+'.dat' ,dtype='int32')
    y=int(len(X)/L**2)
    X=np.reshape(X,(y,L,L))

    for i in range(len(X)):
        m = X[i].sum()/(L**2)
        if m > 0:
            X[i] = - X[i]

    initial_temps = [1.5,1.8,2.0]

    for initial_temp in initial_temps :

        acc = []
        val_acc = []

        initial_index = int((10*(initial_temp-Tmin))*Nconf)

        x1 = X[ initial_index : initial_index + Nconf ]

        for t in temperature:
            index = int(10*(t-Tmin))*Nconf
            x2 = X[ index : index + Nconf ]
            Xconf = np.concatenate((x1,x2))


            print('ylabel.shape ',y_label.shape)
            print('X.shape ',Xconf.shape)

            X_train, X_test, y_train, y_test = train_test_split(Xconf, y_label,test_size=0.15,random_state=302, shuffle=True)
            X_train = tf.expand_dims(X_train, axis=-1)
            X_test = tf.expand_dims(X_test, axis=-1)
            print('shape of X train: ', X_train.shape)
            print('shape of y train: ', y_train.shape)
            print('shape of X test: ', X_test.shape)
            print('shape of y test: ', y_test.shape)


            network = models.Sequential() #initialize the neural network


            network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride, activation='relu', input_shape=input_shape))
            #network.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
            network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride, activation='relu', padding='valid'))



            #dropout: spegne casualmente qualche neurone
            indexlayer = 0
            if dropout:
                network.add(layers.Dropout(dropout_const))
                indexlayer += 1
            network.add(layers.Flatten())



            network.add(layers.Dense(64, activation='relu'))

            network.add(layers.Dense(16, activation='relu'))

            network.add(layers.Dense(2, activation='sigmoid'))



            network.summary() # Shows information about layers and parameters of the entire network


            print(len(network.layers))
            #Train the network

            network.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

            history = network.fit(X_train, y_train, epochs=10000, batch_size=4, validation_data=(X_test, y_test),callbacks = callbacks)# validation_split=0.1)#

            acc.append(history.history["accuracy"][-1])
            val_acc.append(history.history["val_accuracy"][-1])
            print(acc)
            print(val_acc)
            print('end of T = ', t)
        f = open('termo_data/'+str(L)+'/acc_termo_'+str(int(10*initial_temp))+'.txt', "w")
        np.savetxt(f,acc)
        f.close()
        f = open('termo_data/'+str(L)+'/val_acc_termo_'+str(int(10*initial_temp))+'.txt', "w")
        np.savetxt(f,val_acc)
        f.close()



# if (mode == 'Low') and (L==16):
#
#     temperature = ['01','02','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19' ,'20','21', '22', '23' ]
#     acc = []
#     val_acc = []
#     initial_temp = '05'
#
#
#     for t in temperature:
#
#         X=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/termo/SW_LATT_termo_'+initial_temp+'.dat' ,dtype='int32')
#         XX=np.fromfile('/home/riccardo/DOTTORATO/CODE/SW/configurazioni/termo/SW_LATT_termo_'+t+'.dat' ,dtype='int32')
#         X=np.concatenate((X,XX))
#
#         y=int(len(X)/L**2)
#         X=np.reshape(X,(y,L,L))
#
#         for i in range(len(X)):
#             m = X[i].sum()/(L**2)
#             if m > 0:
#                 X[i] = - X[i]
#
#         #norm
#         #X = 0.5*(X+1)
#
#         print('ylabel.shape ',y_label.shape)
#         print('X.shape ',X.shape)
#
#         X_train, X_test, y_train, y_test = train_test_split(X, y_label,test_size=0.15,random_state=302, shuffle=True)
#         X_train = tf.expand_dims(X_train, axis=-1)
#         X_test = tf.expand_dims(X_test, axis=-1)
#         print('shape of X train: ', X_train.shape)
#         print('shape of y train: ', y_train.shape)
#         print('shape of X test: ', X_test.shape)
#         print('shape of y test: ', y_test.shape)
#
#         network = models.Sequential() #initialize the neural network
#
#
#         network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride, activation='relu', input_shape=input_shape))
#         #network.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
#         network.add(layers.Conv2D(Channel, kernel_size=Nf, strides=Stride, activation='relu', padding='valid'))
#
#
#
#         #dropout: spegne casualmente qualche neurone
#         indexlayer = 0
#         if dropout:
#             network.add(layers.Dropout(dropout_const))
#             indexlayer += 1
#         network.add(layers.Flatten())
#
#
#
#         network.add(layers.Dense(64, activation='relu'))
#
#         network.add(layers.Dense(16, activation='relu'))
#
#         network.add(layers.Dense(2, activation='sigmoid'))
#
#
#
#         network.summary() # Shows information about layers and parameters of the entire network
#
#
#         print(len(network.layers))
#
#
#         #Train the network
#
#         network.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
#
#         history = network.fit(X_train, y_train, epochs=10000, batch_size=4, validation_data=(X_test, y_test),callbacks = callbacks)# validation_split=0.1)#
#
#         acc.append(history.history["accuracy"][-1])
#         val_acc.append(history.history["val_accuracy"][-1])
#         print(acc)
#         print(val_acc)
#         print('end of T = ', t)
#
#     # plt.figure(1)
#     # plt.plot(acc)
#     # plt.plot(val_acc)
#     # plt.show()
#     f = open('termo_data/acc_termo_'+initial_temp+'.txt', "w")
#     np.savetxt(f,acc)
#     f.close()
#     f = open('termo_data/val_acc_termo_'+initial_temp+'.txt', "w")
#     np.savetxt(f,val_acc)
#     f.close()
