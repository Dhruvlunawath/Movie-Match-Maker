#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv(r"C:\Users\dhruv\Downloads\archive (3)\TMDB_movie_dataset_v11.csv")
data.head(10)


# In[6]:


data=data[['title', 'vote_average','adult','genres','spoken_languages']]
data.head()


# In[7]:


df = data.iloc[:30000].copy()  


# In[8]:


df.head()


# In[9]:


df['genres'].fillna(df['genres'].mode()[0], inplace=True)
df['spoken_languages'].fillna(df['spoken_languages'].mode()[0], inplace=True)


# In[10]:


df.dropna(subset=['title'], inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer

feature = df["genres"].astype(str).tolist()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)


# In[13]:


from sklearn.decomposition import TruncatedSVD

# Perform TruncatedSVD to reduce dimensionality
svd = TruncatedSVD(n_components=21)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# Compute cosine similarity on the reduced dataset
similarity = cosine_similarity(tfidf_reduced)


# In[15]:


indices = pd.Series(df.index, 
                    index=df['title']).drop_duplicates()


# In[16]:


def match_maker(title, similarity = similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    movieindices = [i[0] for i in similarity_scores]
    return df['title'].iloc[movieindices]
recent_watched=input("Enter recently watched movie:")

print(match_maker(recent_watched))


# In[17]:


import pickle


# In[18]:


pickle.dump(df, open('movies_list.pkl', 'wb'))


# In[19]:


pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[20]:


pickle.load(open('movies_list.pkl', 'rb'))


# In[21]:


# Saving DataFrame to CSV file
df.to_csv('movies.csv', index=False)

