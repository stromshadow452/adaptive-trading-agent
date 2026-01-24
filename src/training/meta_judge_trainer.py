"""
SCOPUS Self-Learning Trainer - MetaJudge Model

This module trains and deploys a MetaJudge model that predicts whether
a trade setup is likely to be "good" or "bad" based on historical patterns.

The MetaJudge acts as a final sanity check before trade execution,
learning from past mistakes to avoid repeating them.

Author: SCOPUS Team
Date: 2025-11-24
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import logging

# Try to import ML libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)


class MetaJudgeTrainer:
    """
    Trains a MetaJudge model to predict trade quality.
    
    The model learns P(good_trade | context_features) and can be used
    to filter out likely bad trades before execution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize MetaJudgeTrainer.
        
        Args:
            config: Configuration dict:
                - model_type: 'lightgbm' or 'xgboost' (default: 'lightgbm')
                - test_size: Test split ratio (default: 0.2)
                - random_state: Random seed (default: 42)
        """
        self.config = config or {}
        
        self.model_type = self.config.get('model_type', 'lightgbm')
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        
        # Validate model type
        if self.model_type == 'lightgbm' and not HAS_LIGHTGBM:
            logger.warning("LightGBM not available, falling back to XGBoost")
            self.model_type = 'xgboost'
        
        if self.model_type == 'xgboost' and not HAS_XGBOOST:
            raise ImportError("Neither LightGBM nor XGBoost available. Install one of them.")
        
        logger.info(f"MetaJudgeTrainer initialized with model_type={self.model_type}")
    
    def prepare_training_data(
        self,
        labeled_trades: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare training dataset from labeled trades.
        
        Args:
            labeled_trades: DataFrame with labeled trades and context
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing training data...")
        
        df = labeled_trades.copy()
        
        # Define feature columns
        feature_cols = []
        
        # Numerical features
        numerical_features = [
            'ml_confidence', 'volatility', 'atr_pct', 'rsi', 'bb_width',
            'hour_of_day', 'day_of_week', 'recent_winrate_10',
            'consecutive_losses', 'bars_since_last_trade',
            'trade_duration_hours'
        ]
        
        for col in numerical_features:
            if col in df.columns:
                feature_cols.append(col)
                # Fill NaN with median
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical features (one-hot encode)
        categorical_features = ['regime', 'side', 'exit_reason']
        
        for col in categorical_features:
            if col in df.columns:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.extend(dummies.columns.tolist())
        
        # Target variable
        y = df['good_trade']
        
        # Features
        X = df[feature_cols]
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train MetaJudge model.
        
        Args:
            X: Feature matrix
            y: Target labels (good_trade)
            feature_names: List of feature names
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        logger.info("Training MetaJudge model...")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Train model
        if self.model_type == 'lightgbm':
            model = self._train_lightgbm(X_train, y_train, X_test, y_test)
        else:
            model = self._train_xgboost(X_train, y_train, X_test, y_test)
        
        # Evaluate
        metrics = self._evaluate(model, X_test, y_test)
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_names)
        metrics['feature_importance'] = feature_importance
        
        logger.info("Training complete!")
        
        return model, metrics
    
    def save_model(
        self,
        model: Any,
        feature_names: List[str],
        metrics: Dict,
        output_path: str
    ):
        """
        Save trained model with metadata.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            metrics: Training metrics
            output_path: Path to save model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Package model with metadata
        package = {
            'model': model,
            'feature_names': feature_names,
            'model_type': self.model_type,
            'metrics': metrics,
            'trained_at': pd.Timestamp.now().isoformat()
        }
        
        joblib.dump(package, output_path)
        logger.info(f"Model saved to {output_path}")
    
    def train_and_save(
        self,
        labeled_trades_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Complete training pipeline: load, prepare, train, save.
        
        Args:
            labeled_trades_path: Path to labeled trades CSV
            output_path: Path to save trained model
            
        Returns:
            Training metrics
        """
        logger.info("=" * 80)
        logger.info("Starting MetaJudge Training")
        logger.info("=" * 80)
        
        # Load data
        df = pd.read_csv(labeled_trades_path, parse_dates=['timestamp_entry', 'timestamp_exit'])
        logger.info(f"Loaded {len(df)} labeled trades")
        
        # Prepare data
        X, y, feature_names = self.prepare_training_data(df)
        
        # Train
        model, metrics = self.train(X, y, feature_names)
        
        # Save
        self.save_model(model, feature_names, metrics, output_path)
        
        # Print summary
        self._print_summary(metrics)
        
        logger.info("=" * 80)
        logger.info("Training Complete")
        logger.info("=" * 80)
        
        return metrics
    
    # ==================== Helper Methods ====================
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model."""
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=50)]
        )
        
        return model
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': self.random_state
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        evals = [(dtrain, 'train'), (dtest, 'test')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=50
        )
        
        return model
    
    def _evaluate(self, model, X_test, y_test) -> Dict[str, Any]:
        """Evaluate model on test set."""
        # Predict
        if self.model_type == 'lightgbm':
            y_pred_proba = model.predict(X_test)
        else:
            dtest = xgb.DMatrix(X_test)
            y_pred_proba = model.predict(dtest)
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'auc': auc,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance."""
        if self.model_type == 'lightgbm':
            importance = model.feature_importance(importance_type='gain')
        else:
            importance = model.get_score(importance_type='gain')
            # Convert to array
            importance = [importance.get(f'f{i}', 0) for i in range(len(feature_names))]
        
        # Normalize
        importance = np.array(importance)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        
        # Create dict
        feature_importance = dict(zip(feature_names, importance.tolist()))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def _print_summary(self, metrics: Dict):
        """Print training summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"AUC:        {metrics['auc']:.4f}")
        logger.info(f"Accuracy:   {metrics['accuracy']:.4f}")
        logger.info(f"Precision:  {metrics['precision']:.4f}")
        logger.info(f"Recall:     {metrics['recall']:.4f}")
        logger.info(f"F1 Score:   {metrics['f1']:.4f}")
        logger.info("")
        logger.info("Top 5 Important Features:")
        for i, (feat, imp) in enumerate(list(metrics['feature_importance'].items())[:5], 1):
            logger.info(f"  {i}. {feat}: {imp:.4f}")
        logger.info("=" * 60)


class MetaJudge:
    """
    Inference wrapper for MetaJudge model.
    
    Predicts P(good_trade | context) for trade filtering.
    """
    
    def __init__(self, model_path: str):
        """
        Load trained MetaJudge model.
        
        Args:
            model_path: Path to saved model (.joblib)
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load package
        package = joblib.load(model_path)
        
        self.model = package['model']
        self.feature_names = package['feature_names']
        self.model_type = package['model_type']
        self.metrics = package.get('metrics', {})
        
        logger.info(f"MetaJudge loaded from {model_path}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"AUC: {self.metrics.get('auc', 'N/A')}")
    
    def score(self, context_features: Dict[str, Any]) -> float:
        """
        Predict probability that trade will be good.
        
        Args:
            context_features: Dict of feature values
            
        Returns:
            Probability [0, 1] that trade is good
        """
        # Build feature vector
        X = self._build_feature_vector(context_features)
        
        # Predict
        if self.model_type == 'lightgbm':
            prob = self.model.predict(X)[0]
        else:
            dmatrix = xgb.DMatrix(X)
            prob = self.model.predict(dmatrix)[0]
        
        return float(prob)
    
    def _build_feature_vector(self, context: Dict[str, Any]) -> pd.DataFrame:
        """Build feature vector from context dict."""
        # Initialize with zeros
        features = {name: 0.0 for name in self.feature_names}
        
        # Fill in provided values
        for key, value in context.items():
            if key in features:
                features[key] = value
            # Handle one-hot encoded features
            elif isinstance(value, str):
                # Check for one-hot columns like 'regime_TREND'
                for feat_name in self.feature_names:
                    if feat_name.startswith(f"{key}_") and feat_name.endswith(f"_{value}"):
                        features[feat_name] = 1.0
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        return X


# ==================== Standalone Usage ====================

if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Train MetaJudge model")
    parser.add_argument('--labeled-trades', required=True, help="Path to labeled trades CSV")
    parser.add_argument('--output', required=True, help="Output path for trained model")
    parser.add_argument('--model-type', default='lightgbm', choices=['lightgbm', 'xgboost'])
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MetaJudgeTrainer(config={'model_type': args.model_type})
    
    # Train and save
    metrics = trainer.train_and_save(
        labeled_trades_path=args.labeled_trades,
        output_path=args.output
    )
    
    print(f"\n✓ Training complete.")
    print(f"✓ Model saved to {args.output}")
    print(f"✓ AUC: {metrics['auc']:.4f}")
