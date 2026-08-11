from fastapi import FastAPI
from src.predict import get_recommendations
import logging
import time

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {
        "message": "Movie Recommendation API"
    }

@app.get("/recommendations/{user_id}")
def recommend(user_id: int):
    logger.info(f"Recommendations endpoint accessed for user: {user_id}")

    start_time = time.time()

    recommendations = get_recommendations(user_id=user_id, top_n=10)

    latency = time.time() - start_time
    logger.info(f"Recommendations generated for user {user_id} in {latency:.4f} seconds")

    return {
        "user_id": user_id,
        "recommendations": recommendations
    }

@app.get("/health")
def health():
    logger.info("Health check endpoint accessed")
    return {
        "Status": "Healthy"
    }