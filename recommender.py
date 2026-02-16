import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, ratings_df, movies_df):
        """
        Initialize recommender with ratings and movies data
        """
        self.ratings_df = ratings_df
        self.movies_df = movies_df

        # Create user-movie matrix
        self.user_movie_matrix = self._create_matrix()

        # Compute similarity between users
        self.similarity_matrix = cosine_similarity(self.user_movie_matrix)

    def _create_matrix(self):
        """
        Create user-movie pivot table
        Rows   -> userId
        Columns-> movieId
        Values -> rating
        """
        matrix = self.ratings_df.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

        return matrix

    def recommend_movies(self, user_id, top_n=5):
        """
        Recommend movies for a given user
        """
        # Check if user exists
        if user_id not in self.user_movie_matrix.index:
            return [{"title": "User not found", "poster": ""}]

        # Find similar users
        user_index = self.user_movie_matrix.index.get_loc(user_id)
        similarity_scores = list(enumerate(self.similarity_matrix[user_index]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top similar users
        similar_users = [i[0] for i in similarity_scores[1:6]]

        # Collect recommended movies
        recommended_movies = set()
        for sim_user in similar_users:
            user_ratings = self.user_movie_matrix.iloc[sim_user]
            liked_movies = user_ratings[user_ratings > 0].index
            recommended_movies.update(liked_movies)

        # Convert movie IDs to titles & posters
        movie_info = self.movies_df.set_index("movieId")

        results = []
        for movie_id in recommended_movies:
            if movie_id in movie_info.index:
                results.append({
                    "title": movie_info.loc[movie_id]["title"],
                    "poster": movie_info.loc[movie_id].get("poster", "")
                })

        return results[:top_n]
