import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

DATA_FILE = "telemetry_data.csv"
MODEL_FILE = "difficulty_model.joblib"

def train_model():
    print("Starting Training Pipeline...")

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Play the game first!")
        return
    
    #1. Load Data
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} game sessions")

    #2. Feature Engineering and Bootstrapping Labels
    # Create a 'target_speed' that our model will learn to predict
    # Rule: If they are highly accurate, speed it up. If they miss it a lot (ball), slow it down
    def calculate_ideal_speed(row):
        current_speed = row['ball_speed_multiplier']
        accuracy = row['player_accuracy']

        if accuracy > 0.75:
            return current_speed * 1.2 # Make it 20% faster than normal speed
        elif accuracy < 0.75:
            return max(0.5, current_speed * 0.8) # Make it 20% slower, min 0.5
        return current_speed # Keeping it as the same
    
    df['target_speed'] = df.apply(calculate_ideal_speed, axis=1)

    # 3. Defining the Features (X) and Target (y)
    # make the model predict the speed the based on how they played
    features = ['player_accuracy', 'avg_reaction_time_ms']
    X = df[features]
    y = df['target_speed']

    # 4. Train the Model
    print("Training Random Forest Model...")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # 5. Save Model
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved successfully to {MODEL_FILE}")


if __name__ == "__main__":
    train_model()


