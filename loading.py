import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle

def load_data(path, img_size, n_classes):
    imgs = []
    labels = []
    ids = []
    cls = []
    for idx in n_classes:
        index = n_classes.index(idx)
        print('Loading {} files (Index: {})'.format(idx, index))
        path_ = os.path.join(path, idx, '*g')
        files = glob.glob(path_)
        for file in files:
            img = cv2.imread(file)
            img = cv2.resize(img, (img_size, img_size), cv2.INTER_LINEAR)
            imgs.append(img)
            label = np.zeros(len(n_classes))
            label[index] = 1.0
            labels.append(label)
            file_base = os.path.basename(file)
            ids.append(file_base)
            cls.append(index)
        imgs = np.array(imgs)
        labels= np.array(labels)
        ids = np.array(ids)
        cls = np.array(cls)
        return imgs, labels, ids, cls



def load_data_test(path, img_size):
    path = os.path.join(path, '*g')
    files = sorted(glob.glob(path))
    X_te = []
    Y_te = []
    print('Reading test images')
    for file in files:
        file_base = os.path.basename(file)
        img = cv2.imread(file)
        img = cv2.resize(img, (img_size, img_size), cv2.INTER_LINEAR)
        X_te.append(img)
        Y_te.append(file_base)
        
    X_te = np.array(X_te, dtype=np.uint8)
    X_te = X_te.astype('float')
    X_te = X_te/255
    
    return X_te, Y_te    




class DataSet(object):
    def __init__(self, img, labels, ids, cls):
        self._num_examples = img.shape[0]
    
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0/255.0)
        self._img = img
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    @property 
    def img(self):
        return self._img
    
    @property 
    def labels(self):
        return self._n_labels
    
    @property 
    def ids(self):
        return self._ids
       
    @property 
    def cls(self):
        return self._cls
    
    @property 
    def num_examples(self):
        return self._num_examples
       
    @property 
    def epochs_completed(self):
        return self._epochs_completed
    
    
def read_data(path, img_size, n_classes, validation=0):
    class Datasets(object):
        pass
    dataset = Datasets()
    
    img, labels, ids, cls = load_data(path, img_size, n_classes)
    img, labels, ids, cls = shuffle(img, labels, ids, cls)
    
    if isinstance(validation, float):
        validation = int(validation * img.shape[0])
    
    X_val = img[:validation]
    Y_val = labels[:validation]
    val_ids = ids[:validation]
    val_cls = cls[:validation]
    
    X_tr = img[validation:]
    Y_tr = labels[validation:]
    tr_ids = ids[validation:]
    tr_cls = ids[validation:]
    
    dataset.train = DataSet(X_tr, Y_tr, tr_ids, tr_cls)
    dataset.valid = DataSet(X_val, Y_val, val_ids, val_cls)
        
        
    return dataset
    
    
def read_data_test(path, img_size):
    img, ids = load_data_test(path, img_size)
    return img, ids
    
    