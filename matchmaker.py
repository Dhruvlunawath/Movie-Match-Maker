import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Load CSV data
df = pd.read_csv(r"C:\Users\dhruv\Downloads\movies.csv")
df=df[:15000].copy()

df['genres'].fillna(df['genres'].mode()[0], inplace=True)
df['spoken_languages'].fillna(df['spoken_languages'].mode()[0], inplace=True)
df.dropna(subset=['title'], inplace=True)

feature = df["genres"].astype(str).tolist()
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)

# Perform TruncatedSVD to reduce dimensionality
svd = TruncatedSVD(n_components=21)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# Compute cosine similarity on the reduced dataset
similarity = cosine_similarity(tfidf_reduced)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, similarity=similarity, indices=indices, df=df):
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    movieindices = [i[0] for i in similarity_scores]
    return df['title'].iloc[movieindices].tolist()


# Streamlit app code
st.header('Movie Match Maker')

movie_list = df['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie)
    
    st.subheader("Recommended Movies:")
    
    for movie in recommended_movie_names[:10]:  
        st.write(movie)


