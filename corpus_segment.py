#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@version: python3.6
@author: vaynetse

"""
import os
import jieba

from Tools import savefile, readfile


def corpus_segment(corpus_path, seg_path):
    '''
    corpus_path is the path for file before division
    seg_path is the path for file after division
    '''
    catelist = os.listdir(corpus_path)  
    '''
    catelist record all the folder names in the corpus_path, including 'art','literature','education'...
   
    '''
    print("the jieba is working")
    # to obtain the file under each folder
    for mydir in catelist:
       
        class_path = corpus_path + mydir + "/"  # train_corpus/art/
        seg_dir = seg_path + mydir + "/"  # train_corpus_seg/art/

        if not os.path.exists(seg_dir):  # create the train_corpus_seg
            os.makedirs(seg_dir)

        file_list = os.listdir(class_path)  
       
   
        for file_path in file_list:  # visit all the file under file_list
            fullname = class_path + file_path  # give the full path：train_corpus/art/21.txt
            content = readfile(fullname)  #read the .txt file
            '''delete the white space,null string,return 
            '''
            content = content.replace('\r\n'.encode('utf-8'), ''.encode('utf-8')).strip()  # delete return
            content = content.replace(' '.encode('utf-8'), ''.encode('utf-8')).strip()  # delete white space
            content_seg = jieba.cut(content)  # 为文件内容分词
            savefile(seg_dir + file_path, ' '.join(content_seg).encode('utf-8')) 
            # put the file after division into seg_path

    print("the division of sentences is finished！！！")


if __name__ == "__main__":
    # division on the training set 
    corpus_path = "./train_corpus/"  
    seg_path = "./train_corpus_seg/" 
    corpus_segment(corpus_path, seg_path)

    # division on the testing set 
    corpus_path = "./test_corpus/"  
    seg_path = "./test_corpus_seg/"  
    corpus_segment(corpus_path, seg_path)
