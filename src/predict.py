import pickle
import pandas as pd

from config import DATA_PATH, MODEL_PATH

movies = pd.read_csv(DATA_PATH / "movies.csv")
ratings = pd.read_csv(DATA_PATH / "ratings.csv")

with open(MODEL_PATH / "movie_recommender.pkl", "rb") as f:
    model = pickle.load(f)

user_id = 1

# Movies already rated by user
rated_movies = set(ratings[ratings["userId"] == user_id]["movieId"])

candidate_movies = movies[~movies["movieId"].isin(rated_movies)]

recommendations = []

for movie_id in candidate_movies["movieId"]:
    predicted_rating = model.predict(uid=user_id, iid=movie_id).est
    recommendations.append((movie_id, predicted_rating))

recommendations.sort(key=lambda x: x[1], reverse=True)
top_10 = recommendations[:10]

print(f"\nTop Recommendations for User {user_id}\n")

for movie_id, score in top_10:
    title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
    print(f"{title} -> {score:.2f}")