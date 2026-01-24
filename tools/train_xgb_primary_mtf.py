"""
Train XGBoost with MTF features

Trains on multi-timeframe features for better predictions.
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
    logger.info("Training XGBoost with MTF features...")
    
    # Load MTF data (M5 base)
    data_file = Path("data/training/eurusd_m5_mtf_features.csv")
    if not data_file.exists():
        logger.error(f"MTF data file not found: {data_file}")
        logger.error("Run: python tools/prepare_training_data_mtf.py first")
        return
    
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Define features - only numeric, exclude target/timestamp
    exclude_cols = ['timestamp', 'target', 'future_return', 'date', 'time']
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    logger.info(f"Using {len(feature_cols)} MTF features")
    logger.info(f"Sample features: {feature_cols[:10]}...")
    
    # Show timeframe distribution
    tf_counts = {}
    for col in feature_cols:
        tf = col.split('_')[0] if '_' in col else 'other'
        tf_counts[tf] = tf_counts.get(tf, 0) + 1
    logger.info(f"Features per timeframe: {tf_counts}")
    
    X = df[feature_cols]
    y = df['target']
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train
    logger.info("Training XGBoost with MTF features...")
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
    
    # Feature importance (top 20)
    logger.info("\nTop 20 MTF features by importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    # Save
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = models_dir / "xgb_primary_mtf.joblib"
    joblib.dump(model, model_file)
    
    logger.info(f"\nModel saved to {model_file}")
    
    # Save feature names
    feature_file = models_dir / "feature_names_mtf.txt"
    with open(feature_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    
    logger.info(f"Feature names saved to {feature_file}")


if __name__ == "__main__":
    main()
