# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:12:08 2019

@author: Danjie
"""

import time
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from pipetools import pipe
import spacy
import collections

class JPD:
    """Job Posting Data object
       file_name should be a string or a list of strings (multiple file names).
       job_field should be a string or a lsit of strings (multiple fields) corresponding to the file_name."""
    def __init__(self, file_name= [], job_field= []):
        if not isinstance(file_name, list):
            file_name= [file_name]
        if not isinstance(job_field, list):
            job_field= [job_field]
        self.file_name= file_name
        self.job_field= job_field
        self.df= self.__ImportData()
        self.df_processed= self.__Preprocess_pipe_01()
        self.df_NP= self.__Preprocess_pipe_02()
        self.df_NP_reduced= self.__RemoveEmptyListRows()
    
    def __ImportData(self):
        """to import job posting text from a list of CSV files"""
        print("Importing files...")
        start_time = time.clock()
        all_data=pd.DataFrame(columns=['sentence', 'field'])
        for file, field in zip(self.file_name, self.job_field):
            print("Importing {} file...".format(field))
            data=pd.read_csv(file, usecols=[7],names=['sentence'], header=None)
            data['field']= self.job_field.index(field)
            print("The {} file contains {} sentences.".format(field, data.shape[0]))
            all_data= all_data.append(data, ignore_index=True)
        print("---This step took {} seconds ---".format(round(time.clock() - start_time, 2)))
        print("The imported data file contains {} sentences.\n".format(all_data.shape[0]))
        return all_data
    
    def __RemoveHyphen(self):
        """to remove meaningless hyphens, like a hyphen at the end of a word or at the end of a sentence, but keep the hyphen between two words"""
        for sent in self.df['sentence']:
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
    
    def __RemoveOtherSymbols(self, series):
        """to remove irrelevant characters, only alphabets, numbers and hyphens left"""
        for sent in series:
            yield re.sub(r'[^A-Za-z0-9-]', ' ', str(sent))
        print("Removing other unintended symbols...\n")
    
    def __get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN) #if not in the dictionary, return NOUN (regard it as noun)

    
    def __Lemmatizer(self, series):
        """Lemmatizing each sentence in the list"""
        lemmatizer= WordNetLemmatizer()
        for sen in series:
            yield ' '.join([lemmatizer.lemmatize(w, self.__get_wordnet_pos(w)) for w in nltk.word_tokenize(sen)])
    
    def __Preprocess_pipe_01(self):
        this_pipe= pipe| self.__RemoveHyphen| self.__RemoveOtherSymbols| self.__Lemmatizer| pd.Series
        df_processed= pd.DataFrame({'sentence': this_pipe(), 'field':self.df['field']})
        return df_processed
#%%
    def __ExtractNP(self):
        """Extracting noun phrases"""
        nlp = spacy.load("en_core_web_sm")
        start_time = time.clock()
        for sent in self.df_processed['sentence']:
            temp= nlp(sent)
            yield [chunk.text for chunk in temp.noun_chunks]
        print("Extracting noun phrases...\n")
        print("---This step took %s seconds ---\n" % round(time.clock() - start_time, 2))


    def __RemovePrimaryStopWords(self, series):
        """Removing primary stop words (the phrases  themselves are stop words)"""
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        for item in series:
            yield [i for i in item if not (i in spacy_stopwords)]
        print("Removing primary stop words in the list...\n")

    def __RemoveSecondaryStopWords(self, series):
        """Removing secondary stop words (those stop words in the phrase)"""
        spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        for item in series:
            temp1=[]
            for it in item:
                temp2 = [i for i in it.split() if not (i in spacy_stopwords)]
                temp1.append(' '.join(temp2))
            yield temp1
        print("Removing secondary stop words in the list...\n")


    def __RemoveEmptyString(self, series):
        """Removing empty strings (i.e., '') in the nested list"""
        for item in series:
            yield [it for it in item if it is not '']
        print("Removing empty strings in the nested list...")


    def __RemoveOneLetterWord(self, series):
        """Remove one letter word (e.g. 's') or empty string (i.e., '') from the nested list"""
        for item in series:
            yield [it for it in item if len(it)>1]
        print("removing one letter words from the nested list...")

    def __ConnectWords(self, series):
        for item in series:
            temp=[]
            for it in item:
                temp.append('_'.join(it.split()))
            yield temp 
        print("Connecting words in the phrase with underscores...\n")
 

    def __Preprocess_pipe_02(self):
        this_pipe= pipe| self.__ExtractNP| self.__RemovePrimaryStopWords| self.__RemoveSecondaryStopWords| self.__RemoveEmptyString| self.__RemoveOneLetterWord| self.__ConnectWords| pd.Series
        df_NP= pd.DataFrame({'sentence': this_pipe(), 'field':self.df_processed['field']})
        return df_NP

#%%
    def __RemoveEmptyListRows(self):
        """Remove rows with empty list (i.e. []) in the nested list"""
        df_NP_reduced= self.df_NP.loc[self.df_NP['sentence'].astype(bool)]
        return df_NP_reduced
        print("Removing rows with empty lists in the nested list...")

#%%

data_file_list=["part-00000-a7a9ded9-bb9d-44cc-8766-c00d59b493e3-c000.csv", "part-00000-bec6d054-899f-4044-ac1d-a2cce46827b7-c000.csv", "part-00000-e52a44cd-6c73-40cf-ac51-7c05caee0942-c000.csv", "part-00000-8c6dc754-753a-4960-abce-ba1bf67ef2c6-c000.csv", "part-00000-e6625ee9-0f56-476f-b170-701cf32fb723-c000.csv"]

data_name_list=["computer science", "human resource management", "geographical engineering", "nursing", "chemical engineering"]            


test1= JPD("part-00000-e52a44cd-6c73-40cf-ac51-7c05caee0942-c000.csv", "geographical engineering")
        
test2= JPD(data_file_list, data_name_list)
