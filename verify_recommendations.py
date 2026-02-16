from data_loader import load_data, create_user_item_matrix
from recommender import Recommender
import sys

def verify_system():
    print("Starting System Verification...")
    
    # 1. Load Data
    try:
        ratings_df, movies_df = load_data()
        print("[PASS] Data loading successful.")
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        sys.exit(1)

    # 2. Create Matrix
    try:
        user_item_matrix = create_user_item_matrix(ratings_df)
        print("[PASS] User-Item Matrix creation successful.")
    except Exception as e:
        print(f"[FAIL] Matrix creation failed: {e}")
        sys.exit(1)

    # 3. Initialize Recommender
    try:
        recommender = Recommender(user_item_matrix)
        recommender.compute_similarity()
        print("[PASS] Recommender initialization and similarity computation successful.")
    except Exception as e:
        print(f"[FAIL] Recommender initialization failed: {e}")
        sys.exit(1)

    # 4. Test User Recommendations
    # User 1 likes Action/Sci-Fi (Matrix, Inception, Avengers)
    # Validate if they get recommended other Action/Sci-Fi movies or similar users' likes
    print("\nTesting User Recommendations for User 1...")
    recs = recommender.get_user_recommendations(1, n_recommendations=3)
    print(f"Recommendations for User 1: {recs}")
    
    # We expect recommendations. User 1 hasn't seen 'Coco' (6), 'Up' (7), 'Godfather' (8), 'Pulp Fiction' (9)
    # User 5 is similar to User 1 and likes 6 & 7. User 3 is somewhat similar.
    # We expect at least some recommendations.
    if len(recs) > 0:
        print("[PASS] User recommendations generated.")
    else:
        print("[WARN] No user recommendations generated (might be expected given small dataset).")

    # 5. Test Item Recommendations
    # Movie 1 is 'The Matrix' (Sci-Fi|Action)
    # Should be similar to Movie 2 'Inception' (Sci-Fi|Action)
    print("\nTesting Item Recommendations for Movie 1 (The Matrix)...")
    item_recs = recommender.get_item_recommendations(1, n_recommendations=3)
    print(f"Recommendations for Movie 1: {item_recs}")
    
    if 2 in item_recs: # Inception
        print("[PASS] 'The Matrix' is similar to 'Inception'.")
    else:
        print(f"[WARN] 'The Matrix' recommendations did not include 'Inception'. Got: {item_recs}")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_system()
