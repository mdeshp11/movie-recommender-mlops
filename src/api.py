from fastapi import FastAPI
from src.predict import get_recommendations

app = FastAPI()


@app.get("/")
def home():

    return {
        "message": "Movie Recommendation API"
    }

@app.get("/recommendations/{user_id}")
def recommend(user_id: int):
    recommendations = get_recommendations(user_id=user_id, top_n=5)
    return {
        "user_id": user_id,
        "recommendations": recommendations
    }