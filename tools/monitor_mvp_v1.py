"""
Monitor MVP v1 performance

Displays feature importance scores and other metrics.
"""

import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("MVP v1 Monitoring Report")
    logger.info("=" * 60)
    
    # Load feature importance
    importance_file = Path("logs/feature_importance.json")
    
    if not importance_file.exists():
        logger.warning(f"Feature importance file not found: {importance_file}")
        logger.info("No trades processed yet with MVP v1")
        return
    
    with open(importance_file, 'r') as f:
        data = json.load(f)
    
    importance_scores = data.get('importance_scores', {})
    batch_count = data.get('batch_count', 0)
    
    logger.info(f"\nBatches processed: {batch_count}")
    logger.info(f"Features tracked: {len(importance_scores)}")
    
    if importance_scores:
        logger.info(f"\nTop 20 features by importance:")
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feat, score) in enumerate(sorted_features[:20], 1):
            logger.info(f"  {i:2d}. {feat:30s}: {score:.4f}")
        
        logger.info(f"\nBottom 10 features:")
        for i, (feat, score) in enumerate(sorted_features[-10:], 1):
            logger.info(f"  {i:2d}. {feat:30s}: {score:.4f}")
    
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
