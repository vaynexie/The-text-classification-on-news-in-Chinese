#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:13:30 2019

@author: vayne
"""


import pickle


# store files
def savefile(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)


# read the file
def readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content


def writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


# read the bunch
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch