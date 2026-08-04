import pickle
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

from config import DATA_PATH, MODEL_PATH

ratings = pd.read_csv(DATA_PATH / "ratings.csv")

reader = Reader(rating_scale=(0.5, 5))

data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD()

model.fit(trainset)

predictions = model.test(testset)

rmse = accuracy.rmse(predictions)

with open(MODEL_PATH / "movie_recommender.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved successfully!")
print(f"RMSE: {rmse}")