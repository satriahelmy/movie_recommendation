import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# Read the movies_metadata file
movies_metadata = pd.read_csv('movies_metadata.csv')

# Load a pre-trained model from Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure the 'overview' column is filled with strings
movies_metadata['overview'] = movies_metadata['overview'].fillna('').astype(str)

# Generate embeddings for each movie overview
tqdm.pandas(desc="Generating embeddings")
movies_metadata['embedding'] = movies_metadata['overview'].progress_apply(lambda x: model.encode(x).tolist())

def recommend_movies(user_input, movies_metadata, model, top_n=5):
    # Generate embedding for the user input
    user_embedding = model.encode(user_input).tolist()
    
    # Calculate cosine similarity between user input embedding and all movie embeddings
    movies_metadata['similarity'] = movies_metadata['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x])[0][0])
    
    # Sort movies by similarity in descending order and get the top n movies
    top_movies = movies_metadata.sort_values(by='similarity', ascending=False).head(top_n)
    
    return top_movies[['title', 'overview', 'similarity']]

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("movies_metadata.pkl", "wb") as file:
    pickle.dump(movies_metadata, file)

# Example usage
user_input = "A story about a young wizard who discovers his magical heritage."
top_n = 5
recommended_movies = recommend_movies(user_input, movies_metadata, model, top_n)
print(recommended_movies)