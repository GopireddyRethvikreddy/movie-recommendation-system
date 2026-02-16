from data_loader import load_data
from recommender import MovieRecommender

def main():
    ratings, movies = load_data("ratings.csv", "movies.csv")
    recommender = MovieRecommender(ratings, movies)

    user_id = int(input("Enter User ID: "))
    recommendations = recommender.recommend_movies(user_id)

    print("Recommended Movies:")
    for movie in recommendations:
        print("-", movie)

if __name__ == "__main__":
    main()
