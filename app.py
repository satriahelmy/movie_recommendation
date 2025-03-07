import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open("movies_metadata.pkl", "rb") as file:
    movies_metadata = pickle.load(file)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Enter a brief description, and we'll recommend similar movies!")

# User input
user_input = st.text_area("Describe a movie plot:")
top_n = st.slider("Number of recommendations:", 1, 10, 5)

# Recommendation function
def recommend_movies(user_input, movies_metadata, model, top_n=5):
    if not user_input:
        return pd.DataFrame(columns=['title', 'overview', 'similarity'])
    
    user_embedding = model.encode(user_input).tolist()
    movies_metadata['similarity'] = movies_metadata['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x])[0][0])
    top_movies = movies_metadata.sort_values(by='similarity', ascending=False).head(top_n)
    return top_movies[['title', 'overview', 'similarity']]

# Button to trigger recommendation
if st.button("Get Recommendations"):
    recommended_movies = recommend_movies(user_input, movies_metadata, model, top_n)
    
    if not recommended_movies.empty:
        for _, row in recommended_movies.iterrows():
            st.subheader(row['title'])
            st.write(row['overview'])
            st.write(f"Similarity Score: {row['similarity']:.4f}")
            st.write("---")
    else:
        st.write("No recommendations found. Try another description!")