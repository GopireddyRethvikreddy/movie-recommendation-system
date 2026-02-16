import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.user_movie_matrix = self._create_matrix()
        self.similarity_matrix = cosine_similarity(self.user_movie_matrix)

    def _create_matrix(self):
        return self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

    def recommend_movies(self, user_id, top_n=5):
        if user_id not in self.user_movie_matrix.index:
            return ["User not found"]

        user_index = self.user_movie_matrix.index.get_loc(user_id)
        similarity_scores = list(enumerate(self.similarity_matrix[user_index]))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        similar_users = [i[0] for i in similarity_scores[1:6]]

        recommended_movies = set()
        for sim_user in similar_users:
            movies = self.user_movie_matrix.iloc[sim_user]
            recommended_movies.update(movies[movies > 0].index)

        movie_titles = self.movies_df.set_index("movieId").loc[list(recommended_movies)]["title"].tolist()
        return movie_titles[:top_n]
# Example usage:
# ratings_df = pd.read_csv('ratings.csv')   