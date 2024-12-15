import streamlit as st
import pandas as pd
import numpy as np
#from streamlit_star_rating import st_star_rating  # Import the star rating component

# Define the base directory where the CSV files are located
base_dir = '/Users/victorananthratchagar/Documents/MCS Course/CS598 - Practical Statistical Learning (PSL)/Project 4/'

# Reading the rating matrix (Rmat.csv)
#rmat = pd.read_csv(f'{base_dir}Rmat.csv')
#rmat = pd.read_csv('https://github.com/victorananth/Movie_Recommendations/blob/main/Rmat_100.csv')
rmat = pd.read_csv('Rmat_100.csv', header=0)

print("Rating Matrix Shape:", rmat.shape)
n_users = rmat.shape[0]
m_movies = rmat.shape[1]

# Getting movie IDs from the rating matrix
m_ids = rmat.columns

# Reading the similarity matrix (similarity.csv)
# similarity_df = pd.read_csv(f'{base_dir}similarity.csv')
similarity_df = pd.read_csv('similarity_top_30_first_100_movies.csv').iloc[:,1:]

similarity_df.columns = m_ids
similarity_df.index = m_ids

# Convert the similarity matrix to a NumPy array
S = similarity_df.values

# Load the pop_ranked.csv, which contains MovieID and UserID columns
pop_ranked = pd.read_csv('pop_ranked_ids.csv')

# Load movie data
movies = pd.read_csv(
    'https://liangfgithub.github.io/MovieData/movies.dat?raw=true',
    sep='::', engine='python', encoding="ISO-8859-1", header=None
)
movies.columns = ['MovieID', 'Title', 'Genres']

# Define the IBCF function to generate movie recommendations
def myIBCF(newuser, S, R, pop_ranked):
    # Identify newuser ratings (non-NaN ratings)
    newuser_ratings_i = np.where(np.isnan(newuser) == False)
    
    # Initialize predicted ratings vector to be populated
    pred_ratings = np.zeros(m_movies)
    
    # Calculate predicted rating for every movie
    for i in range(m_movies):
        denom = np.nansum(S[i][newuser_ratings_i])
        numerator = np.nansum(S[i][newuser_ratings_i] * newuser[newuser_ratings_i])
        if denom == 0:
            pred_ratings[i] = np.nan
        else:
            pred_ratings[i] = numerator / denom
        
    # Get top 10 recommended movie indices
    top_10_i = (-pred_ratings).argsort()[:10]
    top_10 = R.columns[top_10_i]
    
    # For every NaN in the top 10, replace with top movies ranked by popularity
    for i in range(len(top_10)):
        # If an element in the top 10 is not a movie ID (str), replace it
        if type(top_10[i]) != str:
            # Go down the list of movies ranked by popularity (pop_ranked)
            for mid in pop_ranked['MovieID']:
                mid_i = np.where(R.columns == f'm{mid}')
                newuser_rating = newuser[mid_i][0]
                
                # If the new user has not rated it and it's not already in the top 10, add to top 10
                if np.isnan(newuser_rating) and mid not in top_10:
                    top_10[i] = f'm{mid}'
                    break
    return top_10

# Streamlit UI setup
st.title("Movie Recommendation System")

st.write("""
    This is a movie recommendation app using Item-Based Collaborative Filtering (IBCF). 
    Please rate the movies, and we will suggest 10 movies based on your preferences.
""")

st.write("""Developed by Brandon and Victor for PSL - Final Project - Fall 2024""")

# Display a common title for the ratings section
st.subheader("Rate the below movies")

# Map movie IDs to titles
movie_titles = movies.set_index('MovieID')['Title']

# Reset the session state if the reset button is clicked
if 'reset' in st.session_state and st.session_state.reset:
    # Safely delete the session state keys if they exist
    if 'sample_movie_ids' in st.session_state:
        del st.session_state['sample_movie_ids']
    if 'user_ratings' in st.session_state:
        del st.session_state['user_ratings']
    st.session_state.reset = False  # Resetting flag for future resets

# Check if the random movie list is already in session state
if 'sample_movie_ids' not in st.session_state:
    # If not, randomly choose 5 movie IDs and store them in session state
    st.session_state.sample_movie_ids = np.random.choice(rmat.columns, 5, replace=False)

user_ratings = {}

# Create a 5-star rating system where the user clicks the star to rate
for movie_id in st.session_state.sample_movie_ids:
    movie_title = movie_titles[int(movie_id[1:])]  # Get the title using the movie ID (strip 'm' from ID)
    
    # Set the defaultValue to 0 (no stars selected initially)
    rating = st_star_rating(label=movie_title, maxValue=5, defaultValue=0, key=movie_id)
    user_ratings[movie_id] = rating  # Store the rating

# Convert the user ratings to a numpy array (user's ratings vector)
newuser_hyp = np.zeros(m_movies)
newuser_hyp[:] = np.nan
for movie_id, rating in user_ratings.items():
    movie_idx = np.where(m_ids == movie_id)[0][0]
    newuser_hyp[movie_idx] = rating

# Get top 10 recommendations when user clicks the button
if st.button('Get Recommendations'):
    recommendations = myIBCF(newuser_hyp, S, rmat, pop_ranked)
    
    # Display recommended movies
    st.write("Here are your top 10 movie recommendations:")

    # Map recommended movie IDs to the movies DataFrame
    recommended_movie_ids = [int(mid[1:]) for mid in recommendations if isinstance(mid, str)]
    recommended_movies = movies[movies['MovieID'].isin(recommended_movie_ids)]

    # Ensure we display 4 rows with 5 columns
    num_columns = 5
    num_rows = 4

    # Custom CSS to increase vertical space between rows
    st.markdown("""
        <style>
            .row-spacing {
                margin-bottom: 40px; 
            }
        </style>
    """, unsafe_allow_html=True)

    # Loop to create 4 rows of movies
    for row_idx in range(num_rows):
        cols = st.columns(num_columns)
        
        # Loop over the recommended movies and display them
        for i in range(row_idx * num_columns, (row_idx + 1) * num_columns):
            if i >= len(recommended_movies):
                break  # Ensure we don't go beyond the number of recommended movies
            
            row = recommended_movies.iloc[i]
            col_index = i % num_columns  # This ensures the movie is placed in the correct column
            with cols[col_index]:
                # Fetch movie image URL
                image_url = f'https://liangfgithub.github.io/MovieImages/{row["MovieID"]}.jpg'
                
                # Display image and title with fixed width for uniformity
                st.image(image_url, caption=f'{row["Title"]} ({row["Genres"]})', use_container_width=True)
        
        # Add margin between rows by creating a div with the row-spacing class
        if row_idx < num_rows - 1:  # Add space only between rows
            st.markdown('<div class="row-spacing"></div>', unsafe_allow_html=True)

# Add a reset button to reload the movie list
if st.button('Reset Ratings and Reload Movies'):
    st.session_state.reset = True  # Set reset flag
    st.rerun()  # Rerun the app to clear the state
