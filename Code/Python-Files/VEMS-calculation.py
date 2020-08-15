#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all Necessary libraries and packages

import nltk
import string
import json
import pandas
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


"""
Function pre_process()
Parameters- 
    1. corpus: Data that needs to be preprocessed

Preprocessing data to convert all the sentences into lower case, to remove the stop words, non-ascii characters
and punctuation
"""

def pre_process(corpus):
    # convert input corpus to lower case.
    corpus = corpus.lower()
    # collecting a list of stop words from nltk and punctuation form
    # string class and create single array.
    stopset = stopwords.words('english') + list(string.punctuation)
    # remove stop words and punctuations from string.
    # word_tokenize is used to tokenize the input corpus in word tokens.
    corpus = " ".join([i for i in word_tokenize(corpus) if i not in stopset])
    # remove non-ascii characters
    corpus = unidecode(corpus)
    return corpus


# In[3]:


#Lemmatizing the data input

lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None

def lemmatize_sentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)

  res_words = []
  for word, tag in wn_tagged:
    if tag is None:                        
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))

  return " ".join(res_words)


# In[4]:


#Import the data file
with open("TestDf.json", 'r') as f:
    datastore = json.load(f)
    
corpus = []
video_id = []

#Collection of all the captions
for item in datastore:
    corpus.append(item['caption'])


# In[5]:


# sentence pair

#for c in range(len(corpus)):
#    corpus[c] = pre_process(corpus[c])
#    corpus[c] = lemmatize_sentence(corpus[c])
#    print(corpus[c])


# In[6]:


# creating vocabulary using uni-gram and bi-gram
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf_vectorizer.fit(corpus)


# In[38]:


# Importing the two csv files as dataframes (the original and the modified one)
df1 = pandas.read_csv('LongestDf.csv')
df2 = pandas.read_csv('Frames_caption.csv')


# In[39]:


df1 = df1.rename(columns = {"video_id": "vid1"}) 
df1 = df1.rename(columns = {"caption": "cap1"}) 

df2 = df2.rename(columns = {"VideoID": "vid2"}) 
df2 = df2.rename(columns = {"Caption": "cap2"}) 


# In[40]:


print(df1.head())


# In[41]:


print(df2.head())


# In[42]:


#Using the Left-Join function to merge the two dataframes to create a new dataframe named 'merged_left'
N = 150
merged = pandas.merge(left=df2, right=df1, how='left', left_on='vid2', right_on='vid1')
merged


# In[43]:


#To check if there is any row 

merged[ pandas.isnull(merged.vid1)]


# In[44]:


def get_cosine_similarity(feature_vec_1, feature_vec_2):
    return cosine_similarity(feature_vec_1.reshape(1, -1), feature_vec_2.reshape(1, -1))[0][0]


# In[45]:


merged["fsim"] = 0
for i, row in merged.iterrows():
    captions = []
    captions.append(row['cap1'])
    captions.append(row['cap2'])
    
    for c in range(len(captions)):
        captions[c] = pre_process(captions[c])
        captions[c] = lemmatize_sentence(captions[c])
    
    feature_vectors = tfidf_vectorizer.transform(captions)

    fsims = get_cosine_similarity(feature_vectors[0], feature_vectors[1])
    
    merged['FWeight'] = (merged['FEnd'] - merged['FStart'])/merged['Video_Duration']
    merged['fsim'].iloc[i] = fsims


# In[46]:


#print(merged[370:])
df = merged[['vid2', 'COS', 'fsim', 'FWeight']]

print(df)


# In[47]:


df2 = df.groupby('vid2')
similarity=[]

for group in df2:
    similarity.append(np.sum(group[1]['fsim']*group[1]['FWeight'])/ np.sum(group[1]['FWeight']))


# In[48]:


similarity


# In[49]:


VEMS = 0.0

video_scores = []

for i, row in merged.iterrows():
    vid_score = 0.0
    total_weight = 0.0

    for j in range(row['COS']):
        total_weight = total_weight + row['FWeight']
        vid_score = vid_score + (row['FWeight'] * row['fsim'])
    
    i = i + row['COS']
    
    vid_score = vid_score/total_weight
    video_scores.append(vid_score)
    
print(video_scores[:3])


# In[50]:


sumvems = sum(similarity)
VEMS = sumvems / N
print(VEMS)


# In[33]:


# sentence pair
example = ["The man and the dog then starts playing with the ball again.", "A black pet dog dribbles a basketball, catches the ball passed on to him and throws it into the basket."]

for c in range(len(example)):
    example[c] = pre_process(example[c])
    example[c] = lemmatize_sentence(example[c])
    print(example[c])

# creating vocabulary using uni-gram and bi-gram
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf_vectorizer.fit(corpus)

feature_vectors = tfidf_vectorizer.transform(example)

get_cosine_similarity(feature_vectors[0], feature_vectors[1])


# In[ ]:




