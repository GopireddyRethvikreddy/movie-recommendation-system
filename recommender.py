import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Recommender:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        
    def compute_similarity(self):
        """
        Computes the cosine similarity matrix for users and items.
        """
        # User-User Similarity
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        self.user_similarity_df = pd.DataFrame(
            self.user_similarity_matrix, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
        
        # Item-Item Similarity
        # Transpose matrix so items are rows
        item_user_matrix = self.user_item_matrix.T
        self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

    def get_user_recommendations(self, user_id, n_recommendations=3):
        """
        Generates movie recommendations for a specific user based on user similarity.
        Finds similar users, looks at what they liked that the target user hasn't seen.
        """
        if user_id not in self.user_item_matrix.index:
            return []

        # Get top similar users (excluding self)
        similar_users = self.user_similarity_df[user_id].sort_values(ascending=False).index[1:]
        
        recommendations = {}
        
        # Movies the user has already seen
        seen_movies = self.user_item_matrix.loc[user_id]
        seen_movies_ids = seen_movies[seen_movies > 0].index.tolist()
        
        for similar_user in similar_users:
            # Get movies rated by similar user
            sim_user_ratings = self.user_item_matrix.loc[similar_user]
            
            # Filter for movies rated highly (>3), not seen by target user
            candidates = sim_user_ratings[(sim_user_ratings > 3) & (~sim_user_ratings.index.isin(seen_movies_ids))]
            
            for movie_id, rating in candidates.items():
                if movie_id not in recommendations:
                    recommendations[movie_id] = rating
                    
            if len(recommendations) >= n_recommendations:
                break
                
        # Sort by rating (simple approach)
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, rating in sorted_recs[:n_recommendations]]

    def get_item_recommendations(self, movie_id, n_recommendations=3):
        """
        Generates similar movies based on item-item similarity.
        """
        if movie_id not in self.item_similarity_df.index:
            return []
            
        similar_movies = self.item_similarity_df[movie_id].sort_values(ascending=False).index[1:]
        return similar_movies[:n_recommendations].tolist()
