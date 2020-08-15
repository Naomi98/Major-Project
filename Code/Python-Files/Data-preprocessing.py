#!/usr/bin/env python
# coding: utf-8

# In[1]:
## importing necessary packages
import pandas

# In[17]:
## Making a dataframe for the original MSVD dataset

df = pandas.read_csv('video_corpus.csv')
df = df[['VideoID','Start','End','Language','Description']]
df = df.loc[df['Language'] == 'English']
df = df.rename(columns = {"Description": "caption"}) 

# In[19]:
# Extracting features. Making 'video_id' using the features 'VIDEOID', 'Start' and 'End'
df['video_id']=df['VideoID'].astype(str)+'_'+df['Start'].astype(str)+'_'+df['End'].astype(str)
df = df[['video_id','caption']]
print(df.head())

# In[20]:
# Remove NA rows from the data
df = df.dropna()

# In[21]:
# Select one random row for each unique video_id
s = df.index.to_series().groupby(df['video_id']).apply(lambda x: x.sample(n=1))

# In[22]:
# Create a bew dataframe 'canda' using the rows selected above
canda = df.loc[s]

# In[14]:
newdf = df.drop(s)

# In[23]:
# Conversion and saving as csv file
canda.to_csv (r'C:\Users\Vaidehi\Documents\Major Project\video_corpus.csv\RandomA.csv', index = False)

# In[10]:
# Conversion and saving as json file
canda.to_json (r'C:\Users\Vaidehi\Documents\Major Project\CandidateA.json',orient='records')


# In[11]:
sii = newdf.index.to_series().groupby(newdf['video_id']).apply(lambda x: x.sample(n=1))
candb = newdf.loc[sii]

# In[13]:
refdf = newdf.drop(sii)

# In[14]:
candb.to_json (r'C:\Users\Vaidehi\Documents\Major Project\CandidateB.json',orient='records')

refdf.to_json (r'C:\Users\Vaidehi\Documents\Major Project\Referencedf.json',orient='records')

