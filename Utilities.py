# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:55:40 2019

@author: Danjie
"""
#%%
import os
import pandas as pd
import re
import time

def SetDir(Dir):
    print("The default directory is: ", os.getcwd())
    os.chdir(Dir)
    print("The current directory is: ", os.getcwd())
    
#SetDir("F:\BigDataCareer\Python\word2vec")
#os.getcwd()

def ImportData(filename):
    """to import job posting text from a CSV file"""
    print("Importing data...\n")
    start_time = time.clock()
    data=pd.read_csv(filename, usecols=[7],names=['sent'], header=None)
    print("---This step took %s seconds ---\n" % round(time.clock() - start_time, 2))
    print('The sentence list contains %d sentences.' % len(list(data['sent'])))
    return list(data['sent'])

def RemoveHyphen(SentList):
    """to remove meaningless hyphens, like a hyphen at the end of a word or at the end of a sentence, but keep the hyphen between two words"""
    for sent in SentList:
        # remove '---   ' type hyphen
        temp= re.sub(r'\-+\s+', ' ', str(sent))
        # remove '   ---' type hyphen
        temp= re.sub(r'\s+\-+', ' ', temp)
        # remove hyphen at the end
        temp= re.sub(r'\-+$', '', temp)
        # remove hyphen at the beginning
        temp= re.sub(r'^\-+', '', temp)
        yield temp
    print("Removing hyphens...\n")

def RemoveOtherSymbols(SentList):
    """to remove irrelevant characters, only alphabets, numbers and hyphens left"""
    for sent in SentList:
        yield re.sub(r'[^A-Za-z0-9-]', ' ', str(sent))
    print("Removing other unintended symbols...\n")

def SaveDataAsTXT(SentList, ListName):
    """to save a sentence list as a TXT file"""
    print("Saving data as txt file to the working directory...\n")
    with open(ListName, 'w') as f:
        for item in SentList:
            f.write("%s\n" % item)

def ReadTXT(TXTfile):
    """to load a TXT file containing job posting sentences"""
    temp=pd.read_csv(TXTfile, delimiter="\n", header= None)
    return list(temp[0])


