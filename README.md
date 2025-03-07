# Movie Recommendation System

This project is a **Movie Recommendation System** that uses **Sentence Transformers** to generate embeddings from movie overviews and recommends similar movies based on **cosine similarity**.

## Features
- Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) to encode movie overviews into vector embeddings.
- Computes **cosine similarity** between the input description and movie embeddings.
- Provides the **top-N most similar movies** based on the input description.
- Saves and loads the model and processed data using `pickle` for efficiency.

## Installation
Ensure you have Python installed, then install the necessary dependencies:

```bash
pip install pandas numpy sentence-transformers scikit-learn tqdm pickle5 streamlit
```

## Dataset
- **movies_metadata.csv**: Contains metadata about movies, including overviews.
- **ratings.csv**: Contains user ratings (not directly used in this recommendation model).

## Model Training and Embedding Generation
The script:
1. Loads `movies_metadata.csv` and `ratings.csv`.
2. Fills missing values in the `overview` column.
3. Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) to generate embeddings for movie overviews.
4. Stores the processed data in `movies_metadata.pkl` and the transformer model in `model.pkl`.

## Streamlit UI
To run the recommendation system with a simple UI using Streamlit, use:

```bash
streamlit run app.py
```

Ensure you have a script (`app.py`) implementing the **Streamlit UI**.

## Saving and Loading Data
- The model is saved as `model.pkl`.
- The processed movie data (with embeddings) is saved as `movies_metadata.pkl`.

## Notes
- This model uses `all-MiniLM-L6-v2`, which provides fast and efficient embeddings.
- Be sure to **download and preprocess the dataset** before running the recommendation function.

## Future Improvements
- Integrate user ratings to improve recommendations.
- Expand features to include genres, actors, and directors.
- Deploy the application on a web server.

Enjoy using the **Movie Recommendation System**! 🎬🍿
