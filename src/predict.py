import pickle
import pandas as pd

from config import DATA_PATH, MODEL_PATH

# Load data
movies = pd.read_csv(DATA_PATH / "movies.csv")
ratings = pd.read_csv(DATA_PATH / "ratings.csv")

# Load best-performing model
with open(MODEL_PATH / "best_movie_recommender.pkl", "rb") as f:
    model = pickle.load(f)


def get_recommendations(user_id, top_n=10):
    """
    Generate top N movie recommendations for a user.
    """

    # Movies already rated by the user
    rated_movies = set(ratings[ratings["userId"] == user_id]["movieId"])

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
        movie_info = movies.loc[movies["movieId"] == movie_id]

        if movie_info.empty:
            title = f"Unknown Movie ({movie_id})"
            genres= "Unknown Genre"
        else:
            title = movie_info["title"].iloc[0]
            genres= movie_info["genres"].iloc[0]

        results.append({
            "movieId": movie_id,
            "title": title,
            "genres": genres,
            "predicted_rating": score
        })

    return results

def display_recommendations(user_id, top_n=10):
    recommendations = get_recommendations(user_id=user_id, top_n=top_n)

    print("\n" + "=" * 60)
    print(f"Top {top_n} Recommendations for User {user_id}")
    print("=" * 60)

    for movie_id, score in recommendations:
        title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
        print(f"{title} -> Predicted Rating: {score:.2f}")


if __name__ == "__main__":
    user_id = int(input("Enter User ID: "))
    display_recommendations(user_id)