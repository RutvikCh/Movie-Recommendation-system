import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the MovieLens dataset (ratings.csv)
# Assuming you have downloaded the MovieLens dataset and placed it in the same directory as your script.

# Load the ratings data
ratings_data = pd.read_csv('ratings.csv')

# Preview the data
print("Ratings Data:")
print(ratings_data.head())

# Step 2: Create a User-Item Matrix (pivot table)
# This matrix will have rows as users and columns as movies, with ratings as values.
user_item_matrix = ratings_data.pivot_table(index='userId', columns='movieId', values='rating')

# Preview the User-Item Matrix
print("\nUser-Item Matrix:")
print(user_item_matrix.head())

# Step 3: Fill NaN values with 0 (or you can use other techniques like Mean Imputation, but 0 works for this case)
user_item_matrix = user_item_matrix.fillna(0)

# Step 4: Compute Cosine Similarity between users
# Cosine similarity gives a measure of similarity between two vectors (users in this case)
cosine_sim = cosine_similarity(user_item_matrix)

# Step 5: Function to get top N similar users
def get_similar_users(user_id, top_n=5):
    # Get the row corresponding to the user_id
    user_index = user_id - 1  # Since index is 0-based
    similarity_scores = cosine_sim[user_index]
    
    # Create a DataFrame to store user similarity scores
    similarity_df = pd.DataFrame(similarity_scores, index=user_item_matrix.index, columns=['similarity'])
    
    # Sort the users by similarity score in descending order and get the top N most similar users
    similar_users = similarity_df.sort_values(by='similarity', ascending=False).head(top_n)
    
    return similar_users

# Step 6: Make Movie Recommendations
def recommend_movies(user_id, num_recommendations=5):
    # Get similar users to the given user
    similar_users = get_similar_users(user_id, top_n=5)
    
    # Get the movies rated by similar users (excluding the movies already rated by the user)
    similar_users_ratings = user_item_matrix.loc[similar_users.index]
    
    # Calculate the weighted sum of ratings of movies from similar users
    weighted_ratings = similar_users_ratings.mul(similar_users['similarity'], axis=0)
    
    # Sum up the ratings for each movie across similar users
    movie_scores = weighted_ratings.sum(axis=0) / similar_users['similarity'].sum()
    
    # Exclude movies already rated by the user
    movies_rated_by_user = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    movie_scores = movie_scores.drop(movies_rated_by_user, errors='ignore')
    
    # Get the top N recommended movies (highest scores)
    recommended_movies = movie_scores.sort_values(ascending=False).head(num_recommendations)
    
    return recommended_movies

# Step 7: Example of making recommendations for a specific user
user_id = 1  # Example user (you can change it to any user ID)
num_recommendations = 5

recommended_movies = recommend_movies(user_id, num_recommendations)

# Display the recommended movie IDs and their predicted ratings
print(f"\nTop {num_recommendations} Movie Recommendations for User {user_id}:")
print(recommended_movies)

# Optional: Merge with movie titles (if you have movies.csv to get movie titles)
movies = pd.read_csv('movies.csv')

# Merge the movie recommendations with the movie titles
recommended_movie_titles = movies[movies['movieId'].isin(recommended_movies.index)]

# Display the recommended movie titles along with predicted ratings
print("\nMovie Recommendations with Titles:")
for idx, row in recommended_movie_titles.iterrows():
    movie_title = row['title']
    predicted_rating = recommended_movies.loc[row['movieId']]  # Get predicted rating for the movieId
    print(f"{movie_title}: Predicted Rating = {predicted_rating:.2f}")
