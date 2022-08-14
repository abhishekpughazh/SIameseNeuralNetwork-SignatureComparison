import pandas as pd
import sys
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import glob
import cv2
import time
import itertools
import random
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
import tensorflow_hub as hub
from sklearn.utils import shuffle
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image
from PIL.ImageOps import invert


def create_base_network_signet(input_shape):
    '''Base Siamese Network'''
    seq = Sequential()
    seq.add(Conv2D(96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape= input_shape, 
                        kernel_initializer='glorot_uniform', data_format="channels_last"))
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))    
    seq.add(ZeroPadding2D((2, 2), data_format="channels_last"))
    
    seq.add(Conv2D(256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1, kernel_initializer='glorot_uniform', data_format="channels_last"))
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(ZeroPadding2D((1, 1), data_format="channels_last"))
    
    seq.add(Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1, kernel_initializer='glorot_uniform',data_format="channels_last"))
    seq.add(ZeroPadding2D((1, 1), data_format="channels_last"))
    
    seq.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1, kernel_initializer='glorot_uniform',data_format="channels_last"))    
    seq.add(MaxPooling2D((3,3), strides=(2, 2)))
    seq.add(Dropout(0.3))# added extra
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(Dropout(0.5))
    
    seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')) # softmax changed to relu
    
    return seq

root_path = 'F:/GitHub Projects/SiameseNetwork/Dataset'
path1_real= '/dataset1/real/*.*'
path2_real= '/dataset2/real/*.*'
path3_real= '/dataset3/real/*.*'
path4_real= '/dataset4/real/*.*'
path1_forge= '/dataset1/forge/*.*'
path2_forge= '/dataset2/forge/*.*'
path3_forge= '/dataset3/forge/*.*'
path4_forge= '/dataset4/forge/*.*'

gen1=glob.glob(root_path+path1_real)
gen2=glob.glob(root_path+path2_real)
gen3=glob.glob(root_path+path3_real)
gen4=glob.glob(root_path+path4_real)
forg1=glob.glob(root_path+path1_forge)
forg2=glob.glob(root_path+path2_forge)
forg3=glob.glob(root_path+path3_forge)
forg4=glob.glob(root_path+path4_forge)

def reorder(paths):
    index_paths=[]
    for i in range(len(paths)):
        paths_try=paths[i].split('/')
        index_paths.append(int(paths_try[-1][6:8]))
    df_paths=pd.DataFrame(paths,columns=['path'])
    df_paths['index']=index_paths
    df_paths=df_paths.sort_values(by='index')
    return(list(df_paths.iloc[:,0].values))

gen1=reorder(gen1)
forg1=reorder(forg1)
gen2=reorder(gen2)
forg2=reorder(forg2)
gen3=reorder(gen3)
forg3=reorder(forg3)
gen4=reorder(gen4)
forg4=reorder(forg4)

def image_read_pairs(gen,forg):
    #generate the matching label
    left=list()
    right=list()
    for i in range(int(len(gen)/5)):
        person=list()
        for j in range(5*(i),5*(i)+5):
            person.append(gen[j])
        for a in range(len(person)):
            for b in range(a+1,len(person)):
                left.append(person[a])
                right.append(person[b])
    #generate the mismatching label
    for i in range(len(forg)):
        for j in range(5):
            left.append(forg[i])
    for a in range(int(len(gen)/5)):
        for b in range(5):
            for c in range(5*(a),5*(a)+5):
                right.append(gen[c])
    pairs = np.stack([left, right], axis=1)
    return pairs

pair1=image_read_pairs(gen1,forg1)
pair2=image_read_pairs(gen2,forg2)
pair3=image_read_pairs(gen3,forg3)
pair4=image_read_pairs(gen4,forg4)

def generate_label(dataset):
    label_first=list(1 for i in range(len(dataset)*2))
    label_second=list(0 for i in range(len(dataset)*5))
    pair_label=label_first+label_second
    return(pair_label)

label1=generate_label(gen1)
label2=generate_label(gen2)
label3=generate_label(gen3)
label4=generate_label(gen4)

all_pairs= np.concatenate((pair1, pair3), axis=0)
all_label=label1+label3
all_pairs1, all_labels1 = shuffle(all_pairs, all_label)

pairs2,label2 = shuffle(pair2,label2)
pair4,label4=shuffle(pair4,label4)

img_h, img_w = 155, 220

def invert_image_path(path):
    image_file = Image.open(path)  # open colour image
    image_file = image_file.convert('L').resize([220, 155])
    # image_file = image_file.convert('L').resize([224, 224])
    image_file = invert(image_file)
    image_array = np.array(image_file,dtype = np.float64)
    image_array[image_array>=50]=255
    image_array[image_array<50]=0
    return image_array

def generate_batch(all_pairs,all_labels, batch_size = 32):
    '''Function to generate a batch of data with batch_size number of data points
    Half of the data points will be Genuine-Genuine pairs and half will be Genuine-Forged pairs'''
    while True:   
        k = 0
        pairs=[np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]
        targets=np.zeros((batch_size,))
        for ix, pair in enumerate(all_pairs):
            img1 = invert_image_path(pair[0])
            img2 = invert_image_path(pair[1])
            img1 /= 255
            img2 /= 255
            img1 = img1[..., np.newaxis]
            img2 = img2[..., np.newaxis]
            pairs[0][k, :, :, :] = img1
            pairs[1][k, :, :, :] = img2
            targets[k] = all_labels[ix]
            k += 1
            if k == batch_size:
                yield pairs, targets
                k = 0
                pairs=[np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]
                targets=np.zeros((batch_size,))

input_shape= (img_h, img_w, 1)

def euclidean_distance(vects):
    '''Compute Euclidean Distance between two vectors'''
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

data_format="channels_last"

# network definition
base_network = create_base_network_signet(input_shape)

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Compute the Euclidean distance between the two vectors in the latent space
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)

# compile model using RMSProp Optimizer and Contrastive loss function defined above
rms = RMSprop(learning_rate=1e-4, rho=0.9, epsilon=1e-08)
model.compile(loss=contrastive_loss, optimizer=rms, metrics='accuracy')

callbacks = [
    EarlyStopping(patience=12, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    # ModelCheckpoint('signet-bhsig260-{epoch:03d}.h5', verbose=1, save_weights_only=True)
]

batch_sz=10

num_train_samples=len(all_pairs)
num_val_samples=len(pair2)

results = model.fit_generator(generate_batch(all_pairs1,all_labels1, batch_sz),
                              steps_per_epoch = num_train_samples//batch_sz,
                              epochs = 10,
                              validation_data = generate_batch(pair2,label2, batch_sz),
                              validation_steps = num_val_samples//batch_sz,
                              callbacks = callbacks)

model.save('new_model.h5')

def compute_accuracy_roc(predictions, labels):
    '''Compute ROC accuracy with a range of thresholds on distances.
    '''
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
   
    step = 0.01
    max_acc = 0
    best_thresh = -1
   
    for d in np.arange(dmin, dmax+step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel() > d
       
        tpr = float(np.sum(labels[idx1] == 1)) / nsame       
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff
        acc = 0.5 * (tpr + tnr)       
#       print ('ROC', acc, tpr, tnr)
       
        if (acc > max_acc):
            max_acc, best_thresh = acc, d
           
    return max_acc, best_thresh

new_model = model.load_model('new_model.h5', custom_objects={'contrastive_loss': contrastive_loss})


num_test_samples=len(label4)

test_gen = generate_batch(pair4,label4, 1)
pred, tr_y = [], []
for i in range(num_test_samples):
    (img1, img2), label = next(test_gen)
    tr_y.append(label)
    pred.append(model.predict([img1, img2])[0][0])

tr_acc, threshold = compute_accuracy_roc(np.array(pred), np.array(tr_y))
tr_acc, threshold


def predict_score_two(test_gen):
    '''Predict distance score and classify test images as Genuine or Forged'''
    test_point, test_label = next(test_gen)
    img1, img2 = test_point[0], test_point[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
    ax1.imshow(np.squeeze(img1), cmap='gray')
    ax2.imshow(np.squeeze(img2), cmap='gray')
    ax1.set_title('image1')
    ax2.set_title('image2')
    ax1.axis('off')
    ax2.axis('off')
    plt.show()
    result = new_model.predict([img1, img2])
    diff = result[0][0]
    print("Difference Score = ", diff)
    if diff > threshold:
        print("Its a Forged Signature")
    else:
        print("Its a Genuine Signature")

def show_result_two(path1,path2,define_label=0):
    label_test=[define_label]
    left=list()
    right=list()
    left.append(path1)
    right.append(path2)
    pairs = np.stack([left, right], axis=1)
    test_ge = generate_batch(pairs,label_test, 1)
    return predict_score_two(test_ge)

path1 = "/content/03_065.png"
path2 = "/content/04_0118065.PNG"
show_result_two(path1,path2)