import streamlit as st
from data_loader import load_data
from recommender import MovieRecommender

st.set_page_config(page_title="Netflix Recommender", layout="wide")

st.title("🎬 Netflix Style Movie Recommender")

ratings, movies = load_data("ratings.csv", "movies.csv")
recommender = MovieRecommender(ratings, movies)

user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("Recommend"):
    results = recommender.recommend_movies(user_id)

    cols = st.columns(3)
    for i, movie in enumerate(results):
        with cols[i % 3]:
            st.image(movie['poster'], width=150)
            st.write(f"### {movie['title']}")
            st.write("⭐ ⭐ ⭐ ⭐")  # simple stars
            st.subheader("📊 Rating Distribution")

chart_data = ratings['rating'].value_counts().sort_index()
st.bar_chart(chart_data)

