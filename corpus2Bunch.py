#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:58:26 2019

@author: vayne
"""

#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import os
import pickle

from sklearn.datasets.base import Bunch
from Tools import readfile


def corpus2Bunch(wordbag_path, seg_path):
    catelist = os.listdir(seg_path)  # get the category 
    # create a bunch
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(catelist)
  
  
    # obtain the file under path
    for mydir in catelist:
        class_path = seg_path + mydir + "/"  # give the full path 
        file_list = os.listdir(class_path)  # get all files under class_path
        for file_path in file_list:  # visit all the files under path
            fullname = class_path + file_path  
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(readfile(fullname))  #read the txt
            
    # store bunch into the wordbag_path
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("the construction of text object is finished！！！")


if __name__ == "__main__":
    # Bunch on training set：
    wordbag_path = "train_word_bag/train_set.dat" 
    seg_path = "train_corpus_seg/"  
    corpus2Bunch(wordbag_path, seg_path)

     # Bunch on testing set：
    wordbag_path = "test_word_bag/test_set.dat"  
    seg_path = "test_corpus_seg/" 
    corpus2Bunch(wordbag_path, seg_path)
