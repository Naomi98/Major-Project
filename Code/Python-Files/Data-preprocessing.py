#!/usr/bin/env python
# coding: utf-8

# In[1]:


## importing necessary packages
import pandas


# In[17]:


## Making a dataframe for the dataset

df = pandas.read_csv('video_corpus.csv')
df = df[['VideoID','Start','End','Language','Description']]
df = df.loc[df['Language'] == 'English']
print(df)


# In[18]:


df = df.rename(columns = {"Description": "caption"}) 
print(df.head())


# In[19]:


df['video_id']=df['VideoID'].astype(str)+'_'+df['Start'].astype(str)+'_'+df['End'].astype(str)
df = df[['video_id','caption']]
print(df.head())


# In[20]:


#Remove NA rows from the data
df = df.dropna()
print(df)


# In[21]:


#Select one random row for each unique video_id
s = df.index.to_series().groupby(df['video_id']).apply(lambda x: x.sample(n=1))


# In[22]:


canda = df.loc[s]
print(canda[7:80])


# In[14]:


newdf = df.drop(s)


# In[15]:


print(newdf)


# In[23]:


canda.to_csv (r'C:\Users\Vaidehi\Documents\Major Project\video_corpus.csv\RandomA.csv', index = False)


# In[10]:


canda.to_json (r'C:\Users\Vaidehi\Documents\Major Project\CandidateA.json',orient='records')


# In[11]:


sii = newdf.index.to_series().groupby(newdf['video_id']).apply(lambda x: x.sample(n=1))


# In[12]:


candb = newdf.loc[sii]
print(candb)


# In[13]:


refdf = newdf.drop(sii)


# In[14]:


print(refdf)


# In[15]:


candb.to_json (r'C:\Users\Vaidehi\Documents\Major Project\CandidateB.json',orient='records')


# In[17]:


refdf.to_json (r'C:\Users\Vaidehi\Documents\Major Project\Referencedf.json',orient='records')

