import sys

sys.path.insert(0, "/home/lblier/.local/lib/python2.7/site-packages")
sys.path.append("/home/lblier/")

import numpy as np

from os import listdir
from os.path import isfile, join


#from convnets import convnet, preprocess_image_batch
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


import pickle as pkl

import random

from NeuralModels.convnets import convnet, preprocess_image_batch

convnet = convnet('vgg_16',
                  weights_path="../NeuralModels/weights/vgg16_weights.h5")
sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)
convnet.compile(optimizer=sgd, loss='categorical_crossentropy')



y_test = []
with open("ILSVRC2014_clsloc_validation_ground_truth.txt") as f:
    for line in f:
        y_test.append(int(line)-1)

        

def generator_batch(path_img,ground_truth,batch_size=16):
    files = [join(path_img,"ILSVRC2012_val_"+\
                  str(i+1).zfill(8)+".JPEG") \
             for i in range(50000)]
  
    i = 0
    while True:
        X_test = preprocess_image_batch(files[i*batch_size:(i+1)*batch_size],
                                        224,224)
        y_test = ground_truth[i*batch_size:(i+1)*batch_size]
        Y_test = np_utils.to_categorical(y_test, 1000)
        yield X_test,Y_test
        i+=1



gen = generator_batch("/mnt/data/lblier/ImageNet/",
                      y_test,
                      batch_size=16)

# i = 0
# out = []
# truth = []
# while i < 500:
#     print i
#     X,Y = next(gen)
#     out.append(convnet.predict(X))
#     truth.append(Y)
#     i+=1

# out = np.vstack(out)
# classif = out.argmax(axis=1)

# arr_class = [[] for i in range(1000)]
# for i in range(int(classif.shape)):
#     arr_class[classif[i]].append(i)

    

score = convnet.evaluate_generator(generator_batch("/mnt/data/lblier/ImageNet/",
                                                   y_test,
                                                   batch_size=16),
                                   1024,show_accuracy=True,
                                   verbose=1)


print score
