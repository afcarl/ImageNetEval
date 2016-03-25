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

from scipy.io import loadmat

import pickle as pkl

from time import time
from NeuralModels.convnets import convnet, preprocess_image_batch, load_coeff, preprocess_image_batch2

##########################################################
################### PARAMETERS ###########################
meta_clsloc_file = "meta_clsloc.mat"
ground_truth_file = "ILSVRC2014_clsloc_validation_ground_truth.txt"
image_folder = "/mnt/data/lblier/ImageNet/"

batch_size = 16
n_pic = 800

convnet = convnet('alexnet',
                   weights_path="../NeuralModels/weights/alexnet_weights.h5")

###########################################################
###########################################################

sgd = SGD(lr=1., decay=1e-6, momentum=0.9, nesterov=True)
convnet.compile(optimizer=sgd, loss='categorical_crossentropy')

synsets = loadmat(meta_clsloc_file)["synsets"][0][:1000]
synsets = sorted([(int(s[0]), str(s[1][0])) for s in synsets], key=lambda v:v[1])
corr = {}
for i in range(1,1001):
    corr[i] = next(j for j in range(1000) if synsets[j][0] == i)

    

y_test = []
with open(ground_truth_file) as f:
    for line in f:
        y_test.append(corr[int(line)])

        

def generator_batch(path_img,ground_truth,batch_size=16):
    files = [join(path_img,"ILSVRC2012_val_"+\
                  str(i+1).zfill(8)+".JPEG") \
             for i in range(50000)]
  
    i = 0
    while True:
        X_test = preprocess_image_batch2(files[i*batch_size:(i+1)*batch_size])
        y_test = ground_truth[i*batch_size:(i+1)*batch_size]
        Y_test = np_utils.to_categorical(y_test, 1000)
        yield X_test,Y_test
        i+=1



gen = generator_batch(image_folder,
                      y_test,
                      batch_size=batch_size)
i = 0
out = []
truth = []
t0 = time()
while i < n_pic:
    X, Y = next(gen)
    a = convnet.predict(X)
    out.append(a)
    truth.append(Y)

    i+= batch_size
    print i
t1 = time()
out = np.vstack(out)


def top_k(k, out, truth):
    best_k = out.argsort(axis=1)[:,-k:]
    count = sum((truth[j] in best_k[j]) for j in range(n_pic))
    return 1 - float(count) / n_pic

print "Batch size : ", batch_size
print "Number of batches : ", n_pic/batch_size
print "Total time : ", t1 - t0
print "Time per batch : ", (t1 - t0)/(n_pic/batch_size)
print "Time per image  :", (t1 - t0)/n_pic
print "Top 1 error : ", top_k(1, out, y_test)
print "Top 5 error : ", top_k(5, out, y_test)
print "Top 10 error : ", top_k(10, out, y_test)













