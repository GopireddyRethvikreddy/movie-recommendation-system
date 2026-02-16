import pandas as pd
import numpy as np

def load_data():
    """
    Creates a synthetic dataset of users, movies, and ratings.
    Returns:
        ratings_df (pd.DataFrame): DataFrame containing UserID, MovieID, Rating.
        movies_df (pd.DataFrame): DataFrame containing MovieID, Title, Genres.
    """
    # Synthetic Movies Data
    movies_data = {
        'MovieID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Title': [
            'The Matrix', 'Inception', 'Avengers: Endgame', 'Titanic', 'The Notebook',
            'Coco', 'Up', 'The Godfather', 'Pulp Fiction', 'Interstellar'
        ],
        'Genres': [
            'Sci-Fi|Action', 'Sci-Fi|Action', 'Action|Adventure', 'Romance|Drama', 'Romance|Drama',
            'Animation|Adventure', 'Animation|Adventure', 'Crime|Drama', 'Crime|Drama', 'Sci-Fi|Adventure'
        ]
    }
    movies_df = pd.DataFrame(movies_data)

    # Synthetic Ratings Data (Users 1-5)
    # User 1 likes Action/Sci-Fi
    # User 2 likes Romance/Drama
    # User 3 likes Animation
    # User 4 likes Crime/Drama
    # User 5 has mixed tastes corresponding to User 1 and 3
    
    ratings_data = {
        'UserID': [
            1, 1, 1, 1, 
            2, 2, 2, 
            3, 3, 3, 
            4, 4, 
            5, 5, 5, 5
        ],
        'MovieID': [
            1, 2, 3, 10,  # User 1: Matrix, Inception, Avengers, Interstellar
            4, 5, 8,      # User 2: Titanic, Notebook, Godfather
            6, 7, 3,      # User 3: Coco, Up, Avengers
            8, 9,         # User 4: Godfather, Pulp Fiction
            1, 2, 6, 7    # User 5: Matrix, Inception, Coco, Up
        ],
        'Rating': [
            5, 5, 4, 5,
            5, 4, 2,
            5, 5, 3,
            5, 5,
            4, 5, 5, 4
        ]
    }
    ratings_df = pd.DataFrame(ratings_data)
    
    return ratings_df, movies_df

def create_user_item_matrix(ratings_df):
    """
    Creates a user-item matrix from the ratings dataframe.
    """
    user_item_matrix = ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating')
    return user_item_matrix.fillna(0)

if __name__ == "__main__":
    ratings, movies = load_data()
    print("Movies Head:")
    print(movies.head())
    print("\nRatings Head:")
    print(ratings.head())
    
    matrix = create_user_item_matrix(ratings)
    print("\nUser-Item Matrix Shape:", matrix.shape)
