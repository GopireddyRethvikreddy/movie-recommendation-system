import streamlit as st
from data_loader import load_data
from recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Get personalized movie suggestions using AI")

ratings, movies = load_data("ratings.csv", "movies.csv")
recommender = MovieRecommender(ratings, movies)

user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("🎥 Get Recommendations"):
    results = recommender.recommend_movies(user_id)

    if results:
        st.success("Top Recommendations")
        for movie in results:
            st.write("⭐", movie)
    else:
        st.warning("No recommendations found.")
