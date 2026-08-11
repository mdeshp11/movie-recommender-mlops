import pickle
import pandas as pd
from functools import lru_cache

from src.config import DATA_PATH, MODEL_PATH

# Load data
movies = pd.read_csv(DATA_PATH / "movies.csv")
ratings = pd.read_csv(DATA_PATH / "ratings.csv")

# Create a lookup dictionary for movie details for faster access during recommendation generation
movie_lookup = (
    movies
    .set_index("movieId")
    [["title", "genres"]]
    .to_dict("index")
)

user_ratings_cache = (
    ratings
    .groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Load best-performing model
with open(MODEL_PATH / "best_movie_recommender.pkl", "rb") as f:
    model = pickle.load(f)

@lru_cache(maxsize=1000)
def get_recommendations(user_id, top_n=10):
    """
    Generate top N movie recommendations for a user.
    """

    # Movies already rated by the user
    rated_movies = user_ratings_cache.get(user_id, set())

    # Candidate movies = movies user hasn't rated
    candidate_movies = movies[~movies["movieId"].isin(rated_movies)]
    recommendations = []

    for movie_id in candidate_movies["movieId"]:
        predicted_rating = model.predict(
            uid=user_id,
            iid=movie_id
        ).est

        recommendations.append((movie_id, predicted_rating))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    top_recommendations = recommendations[:top_n]
    results = []

    for movie_id, score in top_recommendations:
        movie_data = movie_lookup.get(
            movie_id,
            {
                "title": f"Unknown Movie ({movie_id})",
                "genres": "Unknown Genre"
            }
        )

        results.append(
            {
                "movieId": int(movie_id),
                "title": movie_data["title"],
                "genres": movie_data["genres"],
                "predicted_rating": round(score, 2)
            }
        )

    return results

def display_recommendations(user_id, top_n=10):
    recommendations = get_recommendations(user_id=user_id, top_n=top_n)

    print("\n" + "=" * 80)
    print(f"Top {top_n} Recommendations for User {user_id}")
    print("=" * 80)

    for movie in recommendations:
        print(f"{movie['title']} "
            f"({movie['genres']}) "
            f"-> Predicted Rating: {movie['predicted_rating']}"
        )


if __name__ == "__main__":
    user_id = int(input("Enter User ID: "))
    display_recommendations(user_id)