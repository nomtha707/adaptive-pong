from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import csv
import os
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "difficulty_model.joblib"

# Try to load the model on startup
try:
    model = joblib.load(MODEL_FILE)
    print("✅ ML Model loaded into memory.")
except Exception as e:
    model = None
    print("⚠️ No model found. We will use default difficulty until trained.")

# --- Data Schemas ---
class GameSession(BaseModel):
    session_id: str
    difficulty_level: int
    ball_speed_multiplier: float
    paddle_size_multiplier: float
    player_accuracy: float
    avg_reaction_time_ms: float
    score: int
    misses: int
    session_duration_sec: float

class PlayerStats(BaseModel):
    player_accuracy: float
    avg_reaction_time_ms: float

# --- Endpoints ---
@app.post("/log_session")
async def log_session(session: GameSession):
    file_path = "telemetry_data.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=session.model_dump().keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(session.model_dump())

    return {"status": "success", "message": "Telemetry logged for MLOps pipeline"}

@app.post("/predict_difficulty")
async def predict_difficulty(stats: PlayerStats):
    if model is None:
        # Fallback if no model is trained yet
        return {"suggested_speed_multiplier": 1.0}
    
    # Format data for Scikit-Learn
    features = [[stats.player_accuracy, stats.avg_reaction_time_ms]]
    
    # Get prediction
    predicted_speed = model.predict(features)[0]
    
    # Keep speed within reasonable bounds (e.g., 0.5x to 3.0x)
    safe_speed = max(0.5, min(3.0, predicted_speed))
    
    return {"suggested_speed_multiplier": round(safe_speed, 2)}