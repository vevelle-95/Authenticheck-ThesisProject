import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "6d_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_meta_classifier.json"

def main():
    print("1. Loading 6D Features...")
    # Load the CSV generated in Stage 1
    df = pd.read_csv(FEATURES_PATH)

    # Define the 6 feature columns (X) and the target label (y)
    feature_cols = [
        "dim1_prob_auth", 
        "dim2_prob_dec", 
        "dim3_prob_liv", 
        "dim4_prob_irr", 
        "dim5_clip_sim", 
        "dim6_star_rating"
    ]
    
    X = df[feature_cols]
    # 0 Authentic, 1 Deceptive, 2 Low Informational Value, 3 Irrelevant
    y = df["label_stage1"]

    print("2. Splitting Data...")
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    print("3. Training XGBoost Meta-Classifier...")
    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)

    print("4. Evaluating Model Performance...")
    # Generate predictions on the unseen test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    class_report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=[
            "Authentic (0)",
            "Deceptive (1)",
            "Low Info Value (2)",
            "Irrelevant (3)",
        ],
        zero_division=0,
    )

    print(f"\n--- RESULTS ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # 5. Save the trained model for future use
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print("\nSuccess! Model saved to models/xgboost_meta_classifier.json")

if __name__ == "__main__":
    main()