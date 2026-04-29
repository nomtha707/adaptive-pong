import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
import mlflow
import mlflow.sklearn

DATA_FILE = "telemetry_data.csv"
MODEL_FILE = "difficulty_model.joblib"

def train_model():
    print("🚀 Starting MLOps Training Pipeline...")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found. Play the game first!")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    
    # We need at least a few rows to do a train/test split safely
    if len(df) < 5:
        print("⚠️ Not enough data to train. Play a few more rounds!")
        return
        
    print(f"📊 Loaded {len(df)} game sessions.")

    # 2. Feature Engineering & Bootstrapping Labels
    def calculate_ideal_speed(row):
        current_speed = row['ball_speed_multiplier']
        accuracy = row['player_accuracy']
        if accuracy > 0.75: return current_speed * 1.2
        elif accuracy < 0.40: return max(0.5, current_speed * 0.8)
        return current_speed

    df['target_speed'] = df.apply(calculate_ideal_speed, axis=1)

    # 3. Define Features and Split Data
    features = ['player_accuracy', 'avg_reaction_time_ms']
    X = df[features]
    y = df['target_speed']
    
    # Split: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Set up MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("AdaptivePong_Difficulty_Model")

    with mlflow.start_run():
        print("🧠 Training Random Forest Model...")
        
        # Hyperparameters
        n_trees = 50
        
        # Train Model
        model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate Model
        predictions = model.predict(X_test)
        error = mean_absolute_error(y_test, predictions)
        print(f"🎯 Model Mean Absolute Error (MAE): {error:.4f}")

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", n_trees)
        mlflow.log_metric("mae", error)
        mlflow.log_metric("dataset_size", len(df))
        
        # Log the model itself into MLflow's registry
        mlflow.sklearn.log_model(model, "random-forest-model")

        # 5. Save Model locally for FastAPI to use
        joblib.dump(model, MODEL_FILE)
        print(f"✅ Model saved locally to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
