# recommender.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

class MovieRecommendationSystem:
    def __init__(self, movies_file, ratings_file):
        self.movies_file = movies_file
        self.ratings_file = ratings_file
        self.movies_df = None
        self.ratings_df = None
        self.user_similarity = None
        self.movie_similarity = None
        self.user_item_matrix = None
        self.movie_features = None
    
    def load_data(self):
        """
        Load movie and ratings data
        """
        self.movies_df = pd.read_csv(self.movies_file, encoding="latin1")
        self.ratings_df = pd.read_csv(self.ratings_file, encoding="latin1")
        return f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings"
    
    def preprocess_data(self):
        """
        Preprocess the data for recommendation
        """
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        genres = set()
        for genre_list in self.movies_df['genres'].str.split('|'):
            genres.update(genre_list)
            
        if '(no genres listed)' in genres:
            genres.remove('(no genres listed)')
        
        for genre in genres:
            self.movies_df[genre] = self.movies_df['genres'].apply(lambda x: 1 if genre in x else 0)
        
        self.movie_features = self.movies_df.drop(['movieId', 'title', 'genres'], axis=1)
        return "Data preprocessing complete!"
    
    def build_similarity_matrices(self):
        """
        Build user-user and movie-movie similarity matrices
        """
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.movie_similarity = cosine_similarity(self.movie_features)
        return "Similarity matrices built successfully!"
    
    def get_user_list(self):
        return sorted(self.user_item_matrix.index.tolist())
    
    def get_movie_list(self):
        return [(row['movieId'], row['title']) for _, row in self.movies_df.iterrows()]
    
    def recommend_collaborative_filtering(self, user_id, n=10):
        """
        Generate recommendations based on collaborative filtering
        """
        if user_id not in self.user_item_matrix.index:
            return None, f"User {user_id} not found in the dataset"
        
        user_ratings = self.user_item_matrix.loc[user_id]
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similar_users = self.user_similarity[user_idx]
        
        weighted_ratings = np.zeros(len(self.user_item_matrix.columns))
        
        for i, other_user in enumerate(self.user_item_matrix.index):
            if other_user == user_id:
                continue
            sim_score = similar_users[i]
            if sim_score <= 0.1:
                continue
            other_ratings = self.user_item_matrix.loc[other_user].values
            weighted_ratings += sim_score * other_ratings
        
        movie_ids = self.user_item_matrix.columns
        movie_scores = list(zip(movie_ids, weighted_ratings))
        watched_movies = user_ratings[user_ratings > 0].index
        recs = [(mid, score) for mid, score in movie_scores if mid not in watched_movies and score > 0]
        recs.sort(key=lambda x: x[1], reverse=True)
        
        top_recommendations = recs[:n]
        recommended_movies = []
        for movie_id, score in top_recommendations:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                title = movie_info.iloc[0]['title']
                genres = movie_info.iloc[0]['genres']
                recommended_movies.append({'title': title, 'genres': genres, 'score': score})
        
        return recommended_movies, "Recommendations generated successfully!"
