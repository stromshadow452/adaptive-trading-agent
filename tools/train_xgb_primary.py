"""
Train XGBoost primary model for MVP v1

Trains a binary classifier to predict buy/sell signals.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Training XGBoost primary model...")
    
    # Load data
    data_file = Path("data/training/eurusd_m15_features.csv")
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Run: python tools/prepare_training_data.py first")
        return
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} samples")
    
    # Define features - only use numeric columns
    exclude_cols = ['timestamp', 'target', 'future_return', 'source', 'date', 'time']
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    logger.info(f"Using {len(feature_cols)} numeric features")
    logger.info(f"Features: {feature_cols[:10]}...")  # Show first 10
    
    X = df[feature_cols]
    y = df['target']
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train
    logger.info("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    
    logger.info(f"Train accuracy: {train_acc:.3f}")
    logger.info(f"Val accuracy: {val_acc:.3f}")
    
    logger.info("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred_val))
    
    # Save
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = models_dir / "xgb_primary.joblib"
    joblib.dump(model, model_file)
    
    logger.info(f"Model saved to {model_file}")
    
    # Save feature names
    feature_file = models_dir / "feature_names.txt"
    with open(feature_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    
    logger.info(f"Feature names saved to {feature_file}")


if __name__ == "__main__":
    main()
