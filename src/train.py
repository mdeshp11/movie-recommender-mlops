import pickle
import pandas as pd
import mlflow
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
    {"n_factors" : 60},
    {"n_factors" : 65},
    {"n_factors" : 70},
    {"n_factors" : 75},
    {"n_factors" : 100},
    {"n_factors" : 125},
    {"n_factors" : 150},
    {"n_factors" : 175},
    {"n_factors" : 200}
]

for exp in experiments:

    n_factors = exp["n_factors"]

    with mlflow.start_run(run_name=f"SVD_{n_factors}_factors"):
        model = SVD(n_factors=n_factors, random_state=42)
        model.fit(trainset)
        predictions = model.test(testset)

        rmse = accuracy.rmse(predictions, verbose=False)

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