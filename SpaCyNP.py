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

def RemoveEmptyString(DataList):
    """Removing empty strings (i.e., '') in the nested list"""
    for item in DataList:
        yield [it for it in item if it is not '']
    print("Removing empty strings in the nested list...")

def RemoveOneLetterWord(DataList):
    """Remove one letter word (e.g. 's') or empty string (i.e., '') from the nested list"""
    for item in DataList:
        yield [it for it in item if len(it)>1]
    print("removing one letter words from the nested list...")

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

def RemoveEmptyList(DataList):
    """Remove empty list (i.e. []) in the nested list"""
    for item in DataList:
        if len(item):
            yield item
    print("Removing empty lists in the nested list...")

Job_posting_stop_words= ['experience','ability','work','skill','knowledge','strong','provide','include','application','understand','excellent','ensure','need','new','demonstrate','employee','employer','resume','find','let','appropriate','job','able','opportunity','effective','professional','team','development','support','procedure','project','problem','client','management','system','customer','process','solution','organization','familiarity','implementation','people','level','time','service','business','technology']
def RemoveCustomizedStopWords(DataList):
    for item in DataList:
        yield [it for it in item if not (it in Job_posting_stop_words)]
    print("Removing additional customized stop words from the nested list...")