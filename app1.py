# app.py

import streamlit as st
from recommender import MovieRecommendationSystem

st.title("Movie Recommendation System")

# Load Data
movies_file = r"C:\Users\sahil\OneDrive\Desktop\movies.csv"
ratings_file = r"C:\Users\sahil\OneDrive\Desktop\ratings.csv"

recommender = MovieRecommendationSystem(movies_file, ratings_file)
st.write(recommender.load_data())
st.write(recommender.preprocess_data())
st.write(recommender.build_similarity_matrices())

# User Selection
st.sidebar.header("User Selection")
user_list = recommender.get_user_list()
selected_user = st.sidebar.selectbox("Select User ID", user_list)

# Generate Recommendations
if st.sidebar.button("Get Recommendations"):
    recommendations, msg = recommender.recommend_collaborative_filtering(selected_user, n=10)
    st.sidebar.write(msg)
    if recommendations:
        st.write("### Recommended Movies")
        for rec in recommendations:
            st.write(f"**{rec['title']}** ({rec['genres']}) - Score: {rec['score']:.2f}")
