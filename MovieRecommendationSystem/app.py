from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load datasets
movies = pd.read_csv('movies.csv', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movie_id', 'title'])
ratings = pd.read_csv('ratings.csv', sep='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Create user-item matrix
user_movie_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_movie_recommendations(user_id, num_recommendations=5):
    # Get similarity scores for the user
    sim_scores = user_similarity_df[user_id]
    # Exclude the user itself
    sim_scores = sim_scores.drop(user_id)
    # Get top similar users
    top_users = sim_scores.sort_values(ascending=False).head(10).index
    # Get movies rated by top users
    top_users_ratings = user_movie_matrix.loc[top_users]
    # Compute average ratings for each movie
    avg_ratings = top_users_ratings.mean(axis=0)
    # Exclude movies already rated by the user
    user_rated_movies = user_movie_matrix.loc[user_id]
    movies_to_recommend = avg_ratings[user_rated_movies == 0]
    # Get top N movie IDs
    top_movie_ids = movies_to_recommend.sort_values(ascending=False).head(num_recommendations).index
    # Get movie titles
    recommended_movies = movies[movies['movie_id'].isin(top_movie_ids)]
    return recommended_movies['title'].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = get_movie_recommendations(user_id)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
