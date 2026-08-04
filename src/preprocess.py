import pandas as pd
from config import DATA_PATH

movies = pd.read_csv(DATA_PATH / "movies.csv")
ratings = pd.read_csv(DATA_PATH / "ratings.csv")

print("\nMovies Dataset")
print(movies.head())

print("\nRatings Dataset")
print(ratings.head())

print("\nShape Information")
print("Movies:", movies.shape)
print("Ratings:", ratings.shape)