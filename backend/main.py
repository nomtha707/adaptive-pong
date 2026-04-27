from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import csv
import os

app = FastAPI()

# Allow your local frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data schema
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

@app.post("/log_session")
async def log_session(session: GameSession):
    file_path = "telemetry_data.csv"
    file_exists = os.path.isfile(file_path)

    # Dump raw data to CSV (Our Phase 1 "Database")
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=session.model_dump().keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(session.model_dump())

    return {"status": "success", "message": "Telemetry logged for MLOps pipeline"}