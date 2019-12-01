# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:04:57 2019

@author: Danjie
"""

import spacy
import collections
import time

# load nlp model
nlp = spacy.load("en_core_web_sm")
# load stop words list
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# extract noun phrases
def ExtractNP(DataList):
    """Extracting noun phrases"""
    start_time = time.clock()
    for sent in DataList:
        temp= nlp(sent)
        yield [chunk.text for chunk in temp.noun_chunks]
    print("Extracting noun phrases...\n")
    print("---This step took %s seconds ---\n" % round(time.clock() - start_time, 2))
# remove primary stop words
def RemovePrimaryStopWords(DataList):
    """Removing primary stop words (the phrases  themselves are stop words)"""
    for item in DataList:
        yield [i for i in item if not (i in spacy_stopwords)]
    print("Removing primary stop words in the list...\n")

def RemoveSecondaryStopWords(DataList):
    """Removing secondary stop words (those stop words in the phrase)"""
    for item in DataList:
        temp1=[]
        for it in item:
            temp2 = [i for i in it.split() if not (i in spacy_stopwords)]
            temp1.append(' '.join(temp2))
        yield temp1
    print("Removing secondary stop words in the list...\n")



#connect words in a phrase with underscores
def ConnectWords(DataList):
    for item in DataList:
        temp=[]
        for it in item:
            temp.append('_'.join(it.split()))
        yield temp 
    print("Connecting words in the phrase with underscores...\n")

def WordCounter(WordList, topN):
    """Counting frequency of words or phrases"""
    print("Counting frequency of words or phrases...\n")
    word_counter=collections.Counter(WordList)
    print(word_counter.most_common(topN))
    return word_counter