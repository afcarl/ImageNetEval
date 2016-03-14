from scipy.io import loadmat
import pickle as pkl

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
        except:
            pass

    with open("categories_url.pkl", "wb") as f:
        pkl.dump(dict_class, f)


