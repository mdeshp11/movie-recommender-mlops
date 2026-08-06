import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

from config import DATA_PATH, MODEL_PATH

mlflow.set_experiment("Movie Recommendation System")
ratings = pd.read_csv(DATA_PATH / "ratings.csv")
reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

experiments = [
    {"n_factors" : 50},
    {"n_factors" : 55},
    {"n_factors" : 60},
    {"n_factors" : 65},
    {"n_factors" : 70},
    {"n_factors" : 75},
    {"n_factors" : 80},
    {"n_factors" : 85},
    {"n_factors" : 90},
    {"n_factors" : 95},
    {"n_factors" : 100},
    {"n_factors" : 125},
    {"n_factors" : 150},
    {"n_factors" : 175},
    {"n_factors" : 200}
]

best_rmse = float("inf")
best_model = None
best_n_factors = None

for exp in experiments:

    n_factors = exp["n_factors"]

    with mlflow.start_run(run_name=f"SVD_{n_factors}_factors"):
        model = SVD(n_factors=n_factors, random_state=42)
        model.fit(trainset)
        predictions = model.test(testset)

        rmse = accuracy.rmse(predictions, verbose=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_n_factors = n_factors

        mae = accuracy.mae(
            predictions,
            verbose=False
        )

        mlflow.log_param(
            "n_factors",
            n_factors
        )

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mae",
            mae
        )

        model_name = (f"movie_recommender_{n_factors}.pkl")
        model_path = (MODEL_PATH / model_name)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(str(model_path))
        print(f"SVD({n_factors}) -> RMSE={rmse:.4f}")


best_model_path = MODEL_PATH / "best_movie_recommender.pkl"

with open(best_model_path, "wb") as f:
    pickle.dump(best_model, f)

print("\n" + "=" * 50)
print("BEST MODEL SUMMARY")
print("=" * 50)
print(f"Best n_factors: {best_n_factors}")
print(f"Best RMSE: {best_rmse:.4f}")
print(f"Saved to: {best_model_path}")


with mlflow.start_run(run_name="Best_Model_Summary"):
    mlflow.log_param("best_n_factors", best_n_factors)
    mlflow.log_metric("best_rmse", best_rmse)
    mlflow.log_artifact(str(best_model_path))