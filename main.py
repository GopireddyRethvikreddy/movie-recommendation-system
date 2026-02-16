from data_loader import load_data, create_user_item_matrix
from recommender import Recommender

def main():
    print("Loading data...")
    ratings_df, movies_df = load_data()
    
    print("Creating User-Item Matrix...")
    user_item_matrix = create_user_item_matrix(ratings_df)
    
    print("Initializing Recommender Model...")
    recommender = Recommender(user_item_matrix)
    recommender.compute_similarity()
    
    while True:
        print("\n--- Movie Recommendation System ---")
        print("1. Recommend movies for a User (User-Based Filtering)")
        print("2. Recommend similar movies to a Movie (Item-Based Filtering)")
        print("3. Show Dataset info")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            try:
                user_id = int(input("Enter User ID (1-5): "))
                recs = recommender.get_user_recommendations(user_id)
                if recs:
                    print(f"\nRecommended Movies for User {user_id}:")
                    for movie_id in recs:
                        movie_title = movies_df[movies_df['MovieID'] == movie_id]['Title'].values[0]
                        print(f"- {movie_title}")
                else:
                    print("No recommendations found or User not found.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif choice == '2':
            try:
                print("\nAvailable Movies:")
                print(movies_df[['MovieID', 'Title']].to_string(index=False))
                movie_id = int(input("Enter Movie ID: "))
                recs = recommender.get_item_recommendations(movie_id)
                if recs:
                    print(f"\nMovies similar to '{movies_df[movies_df['MovieID'] == movie_id]['Title'].values[0]}':")
                    for r_id in recs:
                        movie_title = movies_df[movies_df['MovieID'] == r_id]['Title'].values[0]
                        print(f"- {movie_title}")
                else:
                    print("Movie not found.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        elif choice == '3':
            print("\nUsers:", ratings_df['UserID'].unique())
            print("Movies:", movies_df['Title'].unique())
            
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
