
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input, LocallyConnected1D,LocallyConnected2D, Flatten, Dense, Conv1D,Conv2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt


def convert_to_grayscale(images):
    '''
    This function take colored images and convert them to gray scale. 
    This is useful to reduce the dimension of the input for example for the CIFAR10 case.
    '''
    num_channels = images.shape[-1]

    if num_channels == 1:
        # Images are already grayscale
        return images

    if num_channels == 3:
        # RGB to grayscale conversion
        if images.shape[-3] == 3:
            grayscale_images = np.dot(images, [0.2989, 0.5870, 0.1140])
        else:
            grayscale_images = np.dot(images, [0.2989, 0.5870, 0.1140]).transpose(0, 2, 3, 1)
        return grayscale_images

    if num_channels > 3:
        # Assuming the first channel is grayscale and others are additional channels, keep only the first channel
        if images.shape[-3] > 1:
            grayscale_images = images[:, 0]
        else:
            grayscale_images = images[:, 0].transpose(0, 2, 3, 1)
        return grayscale_images

    raise ValueError("Unsupported number of channels in the input images.")

def heaviside(x):
    return 0.5 * (np.sign(x) + 1)


def load_data_MNIST(digit1,digit2,shuffling = True, normalizing = False, norm_mode = 'global'):
    '''
    This function serves to load the MNIST dataset from the file 'mnist.npz',
    that must be on the same directory. Since we are studying binary classification, one has
    to choose a pair of digits. These will be assigned to label '0' and '1', regarding their true
    numerical value.
    '''

    with np.load('mnist.npz', allow_pickle=True) as data:
        X_train = data['x_train'].astype('float')
        y_train = data['y_train']
        X_test = data['x_test'].astype('float')
        y_test = data['y_test']

    #If shuffling, shuffles the dataset
    if shuffling:
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

    
    X_0_train = X_train[y_train  == digit1]
    y_0_train = np.ones(len(X_0_train))-1
    print(f'Train {digit1} shape: {X_0_train.shape}')
    X_1_train = X_train[y_train  == digit2]
    y_1_train = np.ones(len(X_1_train))
    print(f'Train {digit2} shape: {X_1_train.shape}')
    X_0_test = X_test[y_test  == digit1]
    y_0_test = np.ones(len(X_0_test))-1
    print(f'test {digit1} shape: {X_0_test.shape}')
    X_1_test = X_test[y_test  == digit2]
    y_1_test = np.ones(len(X_1_test))
    print(f'test {digit2} shape: {X_1_test.shape}')
    #Balancing
    #This serves the purpose of having a balanced dataset, with equal
    #number of '0' and '1' labelled data
    min_size = min(len(y_1_train),len(y_0_train))
    X_0_train = X_0_train[:min_size]
    X_1_train = X_1_train[:min_size]
    y_0_train = y_0_train[:min_size]
    y_1_train = y_1_train[:min_size]
    print(f'Train {digit1} shape: {X_0_train.shape}, Train {digit2} shape: {X_1_train.shape}')
    min_size = min(len(y_1_test),len(y_0_test))
    X_0_test = X_0_test[:min_size]
    X_1_test = X_1_test[:min_size]
    y_0_test = y_0_test[:min_size]
    y_1_test = y_1_test[:min_size]
    print(f'test {digit1} shape: {X_0_test.shape}, test {digit2} shape: {X_1_test.shape}')

    
    X_train = np.concatenate((X_0_train,X_1_train))
    y_train = np.concatenate((y_0_train,y_1_train))
    
    X_test = np.concatenate((X_0_test,X_1_test))
    y_test = np.concatenate((y_0_test,y_1_test))
    #This normalize the data with particular techniques. For further details please check
    #the 'normalize_data' function. If 'normalizing' is set as False, the images will be simply
    #divided by 255, to make the pixel in the range [0,1]
    if normalizing:
        normalize_data(X_train, norm_mode)
        normalize_data(X_test, norm_mode)
    else:
        X_train = X_train/255
        X_test = X_test/255

    return X_train,  y_train,  X_test,  y_test


def load_data_CIFAR10(digit1,digit2, shuffling = True, normalizing = False, grayscale = False,norm_mode = 'global'):
    '''
    This function serves to load the CIFAR10 dataset from the files in the 
    'cifar-10-batches-py' directory.. Since we are studying binary classification, one has
    to choose a pair of classes. These will be assigned to label '0' and '1'.
    '''
    X_train = np.empty((50000, 3, 32, 32), dtype=float)
    y_train = np.empty((50000,), dtype=np.uint8)

    for i in range(5):
        with open('cifar-10-batches-py/data_batch_' + str(i+1), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            X_train[i*10000:(i+1)*10000, :, :, :] = data[b'data'].astype('float').reshape(-1, 3, 32, 32)
            y_train[i*10000:(i+1)*10000] = np.array(data[b'labels'])

    # Load the test data
    with open('cifar-10-batches-py/test_batch', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        X_test = data[b'data'].astype('float').reshape(-1, 3, 32, 32)
        y_test = np.array(data[b'labels'])

    #If shuffling, shuffles the dataset
    if shuffling:
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        y_train = y_train[indices]

    X_0_train = X_train[y_train  == digit1]
    y_0_train = np.ones(len(X_0_train))-1
    print(f'Train {digit1} shape: {X_0_train.shape}')
    X_1_train = X_train[y_train  == digit2]
    y_1_train = np.ones(len(X_1_train))
    print(f'Train {digit2} shape: {X_1_train.shape}')
    X_0_test = X_test[y_test  == digit1]
    y_0_test = np.ones(len(X_0_test))-1
    print(f'test {digit1} shape: {X_0_test.shape}')
    X_1_test = X_test[y_test  == digit2]
    y_1_test = np.ones(len(X_1_test))
    print(f'test {digit2} shape: {X_1_test.shape}')
    #Balancing
    #This serves the purpose of having a balanced dataset, with equal
    #number of '0' and '1' labelled data
    min_size = min(len(y_1_train),len(y_0_train))
    X_0_train = X_0_train[:min_size]
    X_1_train = X_1_train[:min_size]
    y_0_train = y_0_train[:min_size]
    y_1_train = y_1_train[:min_size]
    print(f'Train {digit1} shape: {X_0_train.shape}, Train {digit2} shape: {X_1_train.shape}')
    min_size = min(len(y_1_test),len(y_0_test))
    X_0_test = X_0_test[:min_size]
    X_1_test = X_1_test[:min_size]
    y_0_test = y_0_test[:min_size]
    y_1_test = y_1_test[:min_size]
    print(f'test {digit1} shape: {X_0_test.shape}, test {digit2} shape: {X_1_test.shape}')

    X_train = np.concatenate((X_0_train,X_1_train))
    y_train = np.concatenate((y_0_train,y_1_train))

    X_test = np.concatenate((X_0_test,X_1_test))
    y_test = np.concatenate((y_0_test,y_1_test))
    if grayscale:
        X_train = convert_to_grayscale(X_train)
        X_test = convert_to_grayscale(X_test)
    #This normalize the data with particular techniques. For further details please check
    #the 'normalize_data' function. If 'normalizing' is set as False, the images will be simply
    #divided by 255, to make the pixel in the range [0,1]
    if normalizing:
        normalize_data(X_train, norm_mode)
        normalize_data(X_test, norm_mode)
    else:
        X_train = X_train/255
        X_test = X_test/255

    return X_train,  y_train,  X_test,  y_test
    

#These are just for the sake of clarity. We show how we compute the internal representation matrix
def kernel_matrix(model,data):
    '''
    This function compute the internal representation matrix in the case
    of a one layer fully-connected neural network. It follows the definition of the text.
    '''
    features = model(data)
    return np.matmul(features, np.transpose(features))/len(features[0])

def kernel_matrix_cnn(model,data,Nc):
    '''
    This function compute the internal representation matrix in the case
    of a one layer convolutaional neural network. It follows the definition of the text.
    '''
    features = model(data)
    return np.einsum('iabc,jabc->ij', features, features)/Nc

def normalize_data(data, norm_mode = 'global'):
    '''
    This implements two ways for normalizing data. With 'global' normalization, we take
    the total mean and standard deviation of the entire dataset and we normalize the data with these
    two numbers using the formula: data -> (data-mean)/std. Every pixel of  is normalized with the same numbers.
    With 'local' normalization every pixel of an image is normalize using the mean and standard deviation
    of that picture, using the same formula of the global normalization.
    '''
    if norm_mode == 'global':
        mean = data.mean()
        std = data.std()
        for i in range(len(data)):
            data[i] = (data[i]-mean)/std 
    elif norm_mode == 'local':
        for i in range(len(data)):
            data[i] = (data[i]-data[i].mean())/data[i].std()


#CLASS
#Here one can choose the wanted classes.
digit1 = 0
digit2 = 1

#ARCHITECTURE
#With this boolean variables one chooses the wanted architecture. 
#conv stand for 2D CNN, conv1d for 1D CNN and fully for FCN
conv = 0
conv1d = 1
fully = 0

#LOSS PARAMETERS
#Here one can choose the wanted threshold for the train_loss. In this way one can 
loss_tresh = 2*10**(-8)
#The simulation will repeat experiments until a training loss less than the 'loss_tresh'.
#The maximum number of trial can be set with the following variable.
max_repetition = 7

#DATASET
#Here one can choose the dataset from 'rand', 'MNIST', 'MNIST_rand', 'CIFAR10', 'CIFAR10_rand'.
#In the case of 'rand' one can choose to label the data using a teacher-student setting,
#by setting the 't_student' variable to True
#One can choose if normalize the data with the function 'normalize_data', by setting to True
#'normalizing_data' and then choosing the wanted mode: 'global' or 'local'.
#If 'normalizing_data' is set to False, in case of dataset with real images, the pixel will be divided by 255.
#'N0' sets the linear size of the input in the case of random data
#'rand_std' sets the standard deviation of the random noise added in the case of
#MNIST_rand and CIFAR10_rand dataset choice.
dataset = 'rand'
t_student = 1
normalizing_data = 1
normalization_mode = 'global'
shuffling_data = 0
grayscale = 1
N0 = 1600
rand_std = 0.2

#ACTIVATION
#Here one can choose to use linear activation or to use particular non-linear activation
#One can choose also if to use biases.
linear = 0
activation = 'tanh'#relu
using_biases = 0

#HYPERPARAMS
#Here one can choose the hyperparameters for the training phase.
dynamics = 'Adam' #gradient descent method
output_dim = 1 #dimension of the output (always set to 1)
kernel_size = 40 #kernel size of the CNN masks
stride = 40 #size of the stride of the CNN masks
LR = 10**-2 #Learning Rate
base_epochs = 2000 #Number of epochs
LambdaV = 10000 #inverse of the standard deviation of the V weights '\lambda_1' in the text
LambdaW = 1  #inverse of the standard deviation of the W weights '\lambda_1' in the text


#EXPERIMENT UTILITES
saving_dir = 'experiments' #saving directory of the result
samples = [1600]  #Here one chooses the size of the trainingset (P in the text)
alpha = np.array([1]) #\alpha = P/N1 or P/Nc like in the test. This togheter with 'samples'
                      #sets the size of the hidden layer
N_models = 1 #This is useful for statistics. One can compute the average of the inner representation
            #matrix over many different trained models.


#This set N0 in the case of real data
if dataset == 'CIFAR10' or dataset == 'CIFAR10_rand':
    N0 = 3*32*32
if dataset == 'MNIST' or dataset == 'MNIST_rand':
    N0 = 28*28


#This is useful for creating comprehensible data file
if linear and fully:
    architecture = 'DL'
elif fully:
    architecture = 'DNL'
elif conv:
    architecture = 'CNN'
elif conv1d:
    architecture = 'CNN1D'



sqrt_N0 = np.sqrt(N0)

#Here the name of the data files is created
name = f'{architecture}_{digit1}{digit2}'
if not linear:
    name = name + '_' + activation
name = name + f'_{dataset}_LV{LambdaV}_LW{LambdaW}_M{N_models}_{dynamics}'

if normalizing_data:
    name = name +'_norm'+'_'+normalization_mode

if 'CIFAR10' in dataset and grayscale:
    name = name + '_gray'



name = name + f'_N0_{N0}'

if t_student:
    name = name + '_tstudent'

if dataset == 'MNIST_rand' or dataset == 'CIFAR10_rand':
    name = name + f'_std{round(rand_std,3)}'


if conv or conv1d:
    name = name + f'_K{kernel_size}_S{stride}'

print(f'Saving the experiment in directory:{name}')


#Take the dataset in the case of real data
if dataset == 'MNIST' or dataset == 'MNIST_rand':
    X_train, y_train,  X_test,  y_test = load_data_MNIST(digit1,digit2,shuffling_data,normalizing_data,normalization_mode)
    print(f'DEBUG: {X_train.shape}')
elif dataset == 'CIFAR10' or dataset == 'CIFAR10_rand':
    X_train, y_train,  X_test,  y_test = load_data_CIFAR10(digit1,digit2,shuffling_data,normalizing_data,grayscale,normalization_mode)

    print(f'DEBUG: {X_train.shape}')

# Create callbacks for early stopping and learning rate reduction
earlystop_callback = {
    'monitor' : 'val_loss',
    'patience' : 50,
    'verbose' : 1,
    'min_delta' : 0.00001,
    #'baseline' : 10**-3
    
}

reduceLR_callback = {
    'monitor' : 'loss',
    'factor' : 0.5,
    'min_delta' : 10**(-13),
    'patience' : 100,#int(base_epochs/2),
    'verbose' : 1,
    'min_lr' : 0.00001
}

early_stop = EarlyStopping(monitor = earlystop_callback['monitor'], 
                            patience = earlystop_callback['patience'],
                            verbose = earlystop_callback['verbose'], 
                            min_delta = earlystop_callback['min_delta'])
reduce_lr = ReduceLROnPlateau(monitor = reduceLR_callback['monitor'],
                            factor = reduceLR_callback['factor'] ,
                            min_delta = reduceLR_callback['min_delta'],
                            patience = reduceLR_callback['patience'],
                            verbose = reduceLR_callback['verbose'],
                            min_lr = reduceLR_callback['min_lr'])


#Take the previously chosen dynamics for gradient descent
if dynamics == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
elif dynamics == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)




#Create directories for saving the data of the experiment
try:
    os.mkdir(f'{saving_dir}/{name}')
except:
    pass
try:
    os.mkdir(f'{saving_dir}/{name}/history')
except:
    pass
try:
    os.mkdir(f'{saving_dir}/{name}/kernel')
except:
    pass





for sample in samples: #We repeat the experiment for every P size chosen

    epochs = base_epochs


    if dataset != 'rand':
        #idx serves to take exactly P samples for the training
        idx = [i for i in range(int(sample/2))]
        for i in range(int(len(X_train)/2),int(len(X_train)/2) + int(sample/2)):
            idx.append(i)
        
        X_train_tmp = X_train[idx]
        y_train_tmp = y_train[idx]

        #Add Gaussian noise if wanted
        if dataset == 'MNIST_rand':
            X_train_tmp += np.random.normal(loc = 0.0, scale = rand_std,   size = (sample,28,28))
        if dataset == 'CIFAR10_rand':
            X_train_tmp += np.random.normal(loc = 0.0, scale = rand_std,   size = (sample,3,32,32))

    elif dataset == 'rand':
        #Create the random data trainingset
        X_train_tmp = np.random.normal(loc = 0, scale = 1.0,   size = (sample,int(np.sqrt(N0)),int(np.sqrt(N0))))
        if t_student:
            #In the teacher student setting, set the label following the sum of
            #the entries of each data in the dataset, as explained in the text
            sums = np.sum(X_train_tmp, axis=(-2, -1))
            y_train_tmp = np.heaviside(sums,0)
            # Get indices that sort the labels
            sorted_indices = np.argsort(y_train_tmp)
            # Order X_train and y_train_tmp based on sorted indices
            X_train_tmp = X_train_tmp[sorted_indices]
            y_train_tmp = y_train_tmp[sorted_indices]
            N_ones = np.sum(y_train_tmp)
        else:
            #If not t-student setting, take random labels
            y_train_tmp = np.concatenate((np.zeros(int(sample/2)),np.ones(int(sample/2))))

    print(y_train_tmp[:10],y_train_tmp[-10:])

    print(f'X_train.shape: {X_train_tmp.shape}, y_train.shape: {y_train_tmp.shape}')

    #'widths' contains the sizes of the hidden layer
    widths = (sample/alpha).astype('int')
    np.save(f'{saving_dir}/{name}/widths_{sample}.npy',widths)

    #iterate the experiment for every widths
    for width in widths:
        if t_student:
            np.save(f'{saving_dir}/{name}/N_ones_{sample}_{width}.npy',N_ones)
        sqrt_N1 = np.sqrt(width)
        print(f'N: {width}, P: {sample}')
        
        before_kernels = []
        after_kernels = []
        #Now the experiment is performed dependently on the chosen architecture.
        #       
        if fully:
            #In the case of FCN the input is vectorized to be 1D
            X_train_tmp = X_train_tmp.reshape((sample,-1))

            sqrt_N0 = np.sqrt(X_train_tmp.shape[-1])
            print(f'N0 = {X_train_tmp.shape[-1]}')
            #Compute and save the covariance matrix for further analysis
            covariance_matrix = np.matmul(X_train_tmp, np.transpose(X_train_tmp))/(len(X_train_tmp[0]))
            np.save(f'{saving_dir}/{name}/kernel/Covariance_{sample}.npy',covariance_matrix)
            #Set the wanted activation function
            if linear:
                funcHidden = lambda x: x/sqrt_N0               
            elif activation == 'tanh':
                funcHidden = lambda x: tf.tanh(x/sqrt_N0)
            elif activation == 'relu':
                funcHidden = lambda x: tf.nn.relu(x/sqrt_N0)
            #The last layer is linearly activated, as explained in the text
            funcLast = lambda x: x/sqrt_N1

            #Iterate for 'N_models' to get statistics
            for i_model in range(N_models):
                
                train_stop = False
                #The training phase stops when the wanted threshold is reached or
                #'max_repetition' is achieved
                while not train_stop:
                    if dynamics == 'SGD':
                        optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
                    elif dynamics == 'Adam':
                        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
                    #'model' is the neural network model
                    model = Sequential()
                    model.add(Dense(width,
                                    activation = funcHidden,
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1./np.sqrt(LambdaW)),#/np.sqrt(784)),
                                    #kernel_regularizer= tf.keras.regularizers.L2(0.001),
                                    use_bias = using_biases,
                                    input_shape=X_train_tmp[0].shape
                                    )
                            )
                    model.add(Dense(output_dim,
                                    activation = funcLast,
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1./np.sqrt(LambdaV)),#/np.sqrt(width)),
                                    #kernel_regularizer= tf.keras.regularizers.L2(0.001),
                                    use_bias = using_biases,
                                    )
                            )
                    #The model is created, the loss function will be computed as MSE
                    model.compile(loss='mean_squared_error',
                                optimizer=optimizer,#tf.keras.optimizers.Adam(learning_rate=LR),
                                metrics=['accuracy'])

                    #'features' is the hidden layer representation
                    features = model.layers[0](X_train_tmp)
                    #'before_kernel' is the internal representation matrix before training, at initialization
                    #this by definition is the NNGP kernel
                    before_kernels.append(np.matmul(features, np.transpose(features))/len(features[0]))

                    first_loss, first_accuracy = model.evaluate(X_train_tmp,y_train_tmp, verbose = 0)
                    loss = loss_tresh + 1

                    cnt = 0

                    while loss > loss_tresh:
                        cnt+=1
                        # Train the model on the dataset.
                        #One can employ early_stop if needed
                        history = model.fit(X_train_tmp,
                                            y_train_tmp,
                                            epochs=epochs, 
                                            batch_size = sample,
                                            #validation_data = (X_test[::2],y_test[::2]),
                                            callbacks=[reduce_lr],#,early_stop],#],
                                            verbose = 0
                                            )
                        loss, accuracy = model.evaluate(X_train_tmp,y_train_tmp, verbose = 0)
                        print(f' iter {cnt} ({epochs} epochs), loss {loss}, acc {accuracy}')

                        if cnt > max_repetition and loss > loss_tresh:
                            #if we have reached the maximum number of repetition and
                            #the loss is still too high exit from the loop and end the experiment
                            loss = loss_tresh - 1


                    if cnt <= max_repetition:
                        print('#########################################')
                        print(f'P: {sample}, W: {width}')
                        print('Pre-train:')
                        print(f"{i_model} model, Acc: {first_accuracy}, loss: {first_loss}")
                        print(f'Post-train after {epochs*cnt}  epochs')
                        print(f"{i_model} model, Acc: {accuracy}, loss: {loss}")
                        print('#########################################')
                        if not i_model:
                            #save an example of the hystory of the training, only for the very first experiment
                            np.save(f'{saving_dir}/{name}/history/{sample}_{width}_history.npy',history.history)

                        features = model.layers[0](X_train_tmp)
                        #compute the internal representation matrix after a successfull training
                        after_kernels.append(np.matmul(features, np.transpose(features))/len(features[0]))

                        sys.stdout.flush()

                        train_stop = True
                    else:
                        #if the maximum number of repetition is reached the experiment is discarded and
                        #restarted
                        before_kernels.pop(-1)
                        print(f'Got stuck in a local minimun...  Restart this experiment!')

        #Here the experiment follows the same strategy of the FCN case, for further details
        #please have a look to the 'fully' case
        elif conv1d:

            X_train_tmp = X_train_tmp.reshape((sample,-1))
            X_train_tmp = np.expand_dims(X_train_tmp, axis=-1)
            print('Train shape: ',X_train_tmp.shape)

            Nc = width*((X_train_tmp[0].shape[0]-kernel_size)/stride)**2
            print(f'Flatten hidden layer size: {Nc}')
            sqrt_Nc = np.sqrt(Nc)
            sqrt_N0 = np.sqrt(kernel_size)

            if linear:
                funcHidden = lambda x: x/sqrt_N0               
            elif activation == 'tanh':
                funcHidden = lambda x: tf.tanh(x/sqrt_N0)
            elif activation == 'relu':
                funcHidden = lambda x: tf.nn.relu(x/sqrt_N0)
        
            for i_model in range(N_models):
                train_stop = False

                while not train_stop:
                    if dynamics == 'SGD':
                        optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
                    elif dynamics == 'Adam':
                        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
                    model = Sequential()
                    model.add(Conv1D(filters=width,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    input_shape=X_train_tmp[0].shape,
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1./np.sqrt(LambdaW)),
                                    activation = funcHidden,
                                    #kernel_regularizer= l2(0.1)
                                    use_bias = using_biases,
                                    )
                            )
                    output_shape = model.layers[0].compute_output_shape(X_train_tmp[0].shape)
                    Nc = int(tf.reduce_prod(output_shape[:]))
                    sqrt_Nc = np.sqrt(Nc)
                    print(f'REAL Flatten hidden layer size: {Nc}')

                    model.add(Flatten())
                    model.add(Dense(output_dim,
                                    activation = lambda x: x/sqrt_Nc,
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1./np.sqrt(LambdaV)),#/np.sqrt(width)),
                                    #kernel_regularizer= tf.keras.regularizers.L2(0.001),
                                    use_bias = using_biases,
                                    )
                            )
                    model.compile(loss='mean_squared_error',
                                optimizer=optimizer,#tf.keras.optimizers.Adam(learning_rate=LR),

                                metrics=['accuracy'])

                    model.summary()
                    features = model.layers[0](X_train_tmp)
                    print('Feature shape: ', features.shape)
                    before_kernels.append(np.einsum('iab,jab->ij', features, features)/Nc)
                    #NNGP.append(np.einsum('ika,jla->ijkl', features, features)/Nc)
                    #print('NNGP shape: ' ,NNGP[0].shape)

                    first_loss, first_accuracy = model.evaluate(X_train_tmp,y_train_tmp, verbose = 0)
                    loss = loss_tresh + 1


                    cnt = 0

                    while loss > loss_tresh:
                        cnt+=1
                            # Train the model on the dataset
                        history = model.fit(X_train_tmp,
                                            y_train_tmp,
                                            epochs=epochs, 
                                            batch_size = sample,
                                            #validation_data = (X_test[::2],y_test[::2]),
                                            callbacks=[reduce_lr],#,early_stop],#],
                                            verbose = 0
                                            )
                        loss, accuracy = model.evaluate(X_train_tmp,y_train_tmp, verbose = 0)
                        print(f' iter {cnt} ({epochs} epochs), loss {loss}, acc {accuracy}')

                        if cnt > max_repetition and loss > loss_tresh:
                            loss = loss_tresh - 1

                    if cnt <= max_repetition:
                        print('#########################################')
                        print(f'P: {sample}, W: {width}')
                        print('Pre-train:')
                        print(f"{i_model} model, Acc: {first_accuracy}, loss: {first_loss}")
                        print(f'Post-train after {epochs*cnt}  epochs')
                        print(f"{i_model} model, Acc: {accuracy}, loss: {loss}")
                        print('#########################################')
                        if not i_model:
                            np.save(f'{saving_dir}/{name}/history/{sample}_{width}_history.npy',history.history)

                        features = model.layers[0](X_train_tmp)
                        after_kernels.append(np.einsum('iab,jab->ij', features, features)/Nc)

                        sys.stdout.flush()

                        train_stop = True
                    else:
                        before_kernels.pop(-1)
                        print(f'Got stuck in a local minimun :( ...  Restart this experiment!')

        elif conv:
            
            if dataset != 'CIFAR10' and dataset != 'CIFAR10_rand':
                X_train_tmp = np.expand_dims(X_train_tmp, axis=-1).astype('float')
            #X_test = np.expand_dims(X_test, axis=-1).astype('float')
            Nc = width*((X_train_tmp[0].shape[0]-kernel_size)/stride)**2
            print(f'Flatten hidden layer size: {Nc}')
            sqrt_Nc = np.sqrt(Nc)
            sqrt_N0 = kernel_size

            if linear:
                funcHidden = lambda x: x/sqrt_N0               
            elif activation == 'tanh':
                funcHidden = lambda x: tf.tanh(x/sqrt_N0)
            elif activation == 'relu':
                funcHidden = lambda x: tf.nn.relu(x/sqrt_N0)
            
            for i_model in range(N_models):
                train_stop = False

                while not train_stop:
                    if dynamics == 'SGD':
                        optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
                    elif dynamics == 'Adam':
                        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
                    model = Sequential()
                    model.add(Conv2D(filters=width,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    input_shape=X_train_tmp[0].shape,
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1./np.sqrt(LambdaW)),
                                    activation = funcHidden,
                                    #kernel_regularizer= l2(0.1)
                                    use_bias = using_biases,
                                    )
                            )
                    output_shape = model.layers[0].compute_output_shape(X_train_tmp[0].shape)
                    Nc = int(tf.reduce_prod(output_shape[:]))
                    sqrt_Nc = np.sqrt(Nc)
                    print(f'REAL Flatten hidden layer size: {Nc}')

                    model.add(Flatten())
                    model.add(Dense(output_dim,
                                    activation = lambda x: x/sqrt_Nc,
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1./np.sqrt(LambdaV)),#/np.sqrt(width)),
                                    #kernel_regularizer= tf.keras.regularizers.L2(0.001),
                                    use_bias = using_biases,
                                    )
                            )
                    model.compile(loss='mean_squared_error',
                                optimizer=optimizer,#tf.keras.optimizers.Adam(learning_rate=LR),

                                metrics=['accuracy'])

                    model.summary()
                    features = model.layers[0](X_train_tmp)
                    print('Feature shape: ', features.shape)
                    before_kernels.append(np.einsum('iabc,jabc->ij', features, features)/Nc)



                    first_loss, first_accuracy = model.evaluate(X_train_tmp,y_train_tmp, verbose = 0)
                    loss = loss_tresh + 1


                    cnt = 0

                    while loss > loss_tresh:
                        cnt+=1
                            # Train the model on the dataset
                        history = model.fit(X_train_tmp,
                                            y_train_tmp,
                                            epochs=epochs, 
                                            batch_size = sample,
                                            #validation_data = (X_test[::2],y_test[::2]),
                                            callbacks=[reduce_lr],#,early_stop],#],
                                            verbose = 0
                                            )
                        loss, accuracy = model.evaluate(X_train_tmp,y_train_tmp, verbose = 0)
                        print(f' iter {cnt} ({epochs} epochs), loss {loss}, acc {accuracy}')

                        if cnt > max_repetition and loss > loss_tresh:
                            loss = loss_tresh - 1


                    if cnt <= max_repetition:
                        print('#########################################')
                        print(f'P: {sample}, W: {width}')
                        print('Pre-train:')
                        print(f"{i_model} model, Acc: {first_accuracy}, loss: {first_loss}")
                        print(f'Post-train after {epochs*cnt}  epochs')
                        print(f"{i_model} model, Acc: {accuracy}, loss: {loss}")
                        print('#########################################')
                        if not i_model:
                            np.save(f'{saving_dir}/{name}/history/{sample}_{width}_history.npy',history.history)

                        features = model.layers[0](X_train_tmp)
                        print(features.shape)

                        after_kernels.append(np.einsum('iabc,jabc->ij', features, features)/Nc)

                        sys.stdout.flush()

                        train_stop = True
                    else:
                        before_kernels.pop(-1)
                        print(f'Got stuck in a local minimun :( ...  Restart this experiment!')


        #At the end, average over N_models and save the results in the correct directories

        before_kernels = np.array(before_kernels)
        np.save(f'{saving_dir}/{name}/kernel/KernelNotTrained_{sample}_{int(width)}_no_v.npy',before_kernels.mean(axis = 0))   
        np.save(f'{saving_dir}/{name}/kernel/KernelNotTrained_{sample}_{int(width)}_no_v_std.npy',before_kernels.std(axis = 0)/np.sqrt(N_models-1))
        after_kernels = np.array(after_kernels)
        np.save(f'{saving_dir}/{name}/kernel/KernelTrained_{sample}_{int(width)}_no_v.npy',after_kernels.mean(axis = 0))   
        np.save(f'{saving_dir}/{name}/kernel/KernelTrained_{sample}_{int(width)}_no_v_std.npy',after_kernels.std(axis = 0)/np.sqrt(N_models-1))
