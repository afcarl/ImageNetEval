from scipy.io import loadmat
import pickle as pkl
from random import shuffle
import os
from os.path import join
from parallel_sync import wget

from urllib import urlretrieve

from PIL import Image
import imghdr
import pdb

def categories_url():
    dict_syn = {}

    with open("fall11_urls.txt",'rb') as f:
        for l in f:
            try:
                synset,url = l.split()
                synset=synset.split('_')[0]
                if not synset in dict_syn:
                    dict_syn[synset] = []
                dict_syn[synset].append(url)
            
            except:
                pass

    matfile = loadmat("../ILSVRC2014_devkit/data/meta_clsloc.mat")["synsets"]
    dict_class = {}
    for i in range(1,1001):
        try:
            dict_class[i] = dict_syn[str(matfile[0,i-1][1][0])]
            shuffle(dict_class[i])
        except:
            pass

    with open("categories_url.pkl", "wb") as f:
        pkl.dump(dict_class, f)
    return dict_class


def testset_url(n_img, path):
    with open("categories_url.pkl", "rb") as f:
        dict_class = pkl.load(f)

    for i in dict_class.keys():
        directory = "class"+str(i)+"/"
        if not os.path.exists(join(path,directory)):
            os.makedirs(join(path,directory))

    dict_start_remaining = dict((k, (0, n_img)) for k in dict_class.keys())
    for i in dict_class.keys():
        directory = "class"+str(i)+"/"
        files = [f for f in os.listdir(join(path, directory))]
        remaining = n_img - len(files)
        s, r = dict_start_remaining[i]
        dict_start_remaining[i] = (s,remaining)

    epoch = 0
    while any(r for (k, (s,r)) in dict_start_remaining.iteritems()):
        print "epoch : ", epoch
        #pdb.set_trace()
        urls = [dict_class[i][j] for j in range(dict_start_remaining[i][0],
                                                sum(dict_start_remaining[i])) \
                for i in dict_class.keys()]
        outs = [join(path,
                     "class"+str(i),
                     "img_"+str(j)+"."+
                     dict_class[i][j].split("/")[-1].split(".")[-1]) \
                for j in range(dict_start_remaining[i][0],
                               sum(dict_start_remaining[i])) \
                for i in dict_class.keys()]

        for s in range(len(urls)/10 +1):
            try:
                wget.download('./', urls[s:s+10],
                              outs[s:s+10], tries=1)
            except:
                pass

        for i in dict_class.keys():
            directory = "class"+str(i)+"/"
            files = [f for f in os.listdir(join(path, directory))]

            for f in files:
                filename = join(path, directory, f)
                try:
                    t = imghdr.what(filename)
                    given_type = filename.split(".")[-1]
                    if t and t == given_type or \
                       (t in ["jpg", "jpeg"] and given_type in ["jpg", "jpeg"]):
                        n_pic_class += 1
                    else:
                        os.remove(filename)
                except:
                    os.remove(filename)

            files = [f for f in os.listdir(join(path, directory))]
            remaining = n_img - len(files)
            s, r = dict_start_remaining[i]
            dict_start_remaining[i] = (s+r,remaining)

        epoch += 1

        
    

            
    # for k,v in dict_class.iteritems():
    #     v = v[:n_img]

    # for i in dict_class.keys():
    #     directory = "class"+str(i)+"/"
    #     print("")
    #     print("Class "+str(i))
    #     if not os.path.exists(join(path,directory)):
    #         os.makedirs(join(path,directory))

    #     n_pic_class = 0
    #     j = 0
    #     while n_pic_class < n_img and j < len(dict_class[i]):
    #         try:
    #             filename = wget.download(dict_class[i][j], out=join(path,directory))
    #             try:
    #                 t = imghdr.what(filename)
    #                 given_type = filename.split(".")[-1]
    #                 if t and t == given_type or \
    #                    (t in ["jpg", "jpeg"] and given_type in ["jpg", "jpeg"]):
    #                     n_pic_class += 1
    #                 else:
    #                     os.remove(filename)
    #             except:
    #                 os.remove(filename)
                                                           
    #         except:
    #             pass
    #         j+=1
    #     if n_pic_class < n_img:
    #         print "Error for class "+str(i)+" : only "+str(n_pic_class)+" pictures"
        


#categories_url()
testset_url(20,"data")
