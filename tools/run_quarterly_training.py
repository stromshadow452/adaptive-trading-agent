"""
SCOPUS Self-Learning Trainer - Quarterly Training Orchestrator

This script orchestrates the complete quarterly self-learning cycle:
1. Load last 3 months of trades
2. Analyze and label trades (PerformanceAnalyzer)
3. Mine edges and mistakes (EdgeAndMistakeMiner)
4. Retrain MetaJudge model
5. (Optional) Fine-tune ML Brain and RL Brain
6. Generate training report

Run this script every ~3 months to improve the agent's decision-making.

Usage:
    python -m tools.run_quarterly_training \
        --symbol EURUSD \
        --from 2023-01-01 \
        --to 2023-03-31 \
        --output-dir models/quarterly_2023Q1

Author: SCOPUS Team
Date: 2025-11-24
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Import training modules
from src.training.performance_analyzer import PerformanceAnalyzer
from src.training.edge_mining import EdgeAndMistakeMiner
from src.training.meta_judge_trainer import MetaJudgeTrainer

# Import history logging
from src.history.training_history import TrainingRun, log_training_run

logger = logging.getLogger(__name__)


class QuarterlyTrainingOrchestrator:
    """
    Orchestrates the complete quarterly self-learning training cycle.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator.
        
        Args:
            config: Configuration dict with paths and parameters
        """
        self.config = config
        
        # Paths
        self.trades_path = config['trades_path']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Date range
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.symbol = config.get('symbol', 'EURUSD')
        
        # Initialize components
        self.analyzer = PerformanceAnalyzer(config.get('analyzer_config', {}))
        self.miner = EdgeAndMistakeMiner(config.get('miner_config', {}))
        self.meta_judge_trainer = MetaJudgeTrainer(config.get('meta_judge_config', {}))
        
        logger.info(f"QuarterlyTrainingOrchestrator initialized for {self.symbol} from {self.start_date} to {self.end_date}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete quarterly training pipeline.
        
        Returns:
            Training report dict
        """
        logger.info("=" * 80)
        logger.info("QUARTERLY SELF-LEARNING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)
        
        report = {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'trained_at': datetime.now().isoformat(),
            'stages': {}
        }
        
        # Stage 1: Load and analyze trades
        logger.info("\n[Stage 1/5] Performance Analysis")
        logger.info("-" * 80)
        labeled_trades_path = self.output_dir / 'labeled_trades.csv'
        labeled_df = self.analyzer.analyze_trades(
            trades_path=self.trades_path,
            features_df=None,  # TODO: Load features if available
            output_path=str(labeled_trades_path)
        )
        
        report['stages']['performance_analysis'] = {
            'total_trades': len(labeled_df),
            'good_trades': int((labeled_df['good_trade'] == 1).sum()),
            'bad_trades': int((labeled_df['good_trade'] == 0).sum()),
            'winrate': float((labeled_df['pnl'] > 0).mean())
        }
        
        # Stage 2: Mine edges and mistakes
        logger.info("\n[Stage 2/5] Edge and Mistake Mining")
        logger.info("-" * 80)
        edges, mistakes = self.miner.analyze_and_export(
            labeled_trades_path=str(labeled_trades_path),
            output_dir=str(self.output_dir)
        )
        
        report['stages']['pattern_mining'] = {
            'edges_found': len(edges),
            'mistakes_found': len(mistakes),
            'top_edge': edges[0]['description'] if edges else None,
            'top_mistake': mistakes[0]['description'] if mistakes else None
        }
        
        # Stage 3: Train MetaJudge
        logger.info("\n[Stage 3/5] MetaJudge Training")
        logger.info("-" * 80)
        meta_judge_path = self.output_dir / 'meta_judge_latest.joblib'
        meta_judge_metrics = self.meta_judge_trainer.train_and_save(
            labeled_trades_path=str(labeled_trades_path),
            output_path=str(meta_judge_path)
        )
        
        report['stages']['meta_judge_training'] = {
            'model_path': str(meta_judge_path),
            'auc': float(meta_judge_metrics['auc']),
            'accuracy': float(meta_judge_metrics['accuracy']),
            'precision': float(meta_judge_metrics['precision']),
            'recall': float(meta_judge_metrics['recall'])
        }
        
        # Stage 4: (Optional) Fine-tune ML Brain
        logger.info("\n[Stage 4/5] ML Brain Fine-tuning (Optional)")
        logger.info("-" * 80)
        if self.config.get('finetune_ml_brain', False):
            logger.info("ML Brain fine-tuning requested but not implemented yet")
            logger.info("TODO: Implement incremental XGBoost training")
            report['stages']['ml_brain_finetuning'] = {'status': 'skipped'}
        else:
            logger.info("ML Brain fine-tuning skipped (set finetune_ml_brain=True to enable)")
            report['stages']['ml_brain_finetuning'] = {'status': 'disabled'}
        
        # Stage 5: (Optional) Fine-tune RL Brain
        logger.info("\n[Stage 5/5] RL Brain Fine-tuning (Optional)")
        logger.info("-" * 80)
        if self.config.get('finetune_rl_brain', False):
            logger.info("RL Brain fine-tuning requested but not implemented yet")
            logger.info("TODO: Implement PPO fine-tuning with recent episodes")
            report['stages']['rl_brain_finetuning'] = {'status': 'skipped'}
        else:
            logger.info("RL Brain fine-tuning skipped (set finetune_rl_brain=True to enable)")
            report['stages']['rl_brain_finetuning'] = {'status': 'disabled'}
        
        # Save training report
        report_path = self.output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log to training history
        try:
            self._log_to_history(report, report_path, meta_judge_path)
        except Exception as e:
            logger.warning(f"Failed to log training to history: {e}")
        
        logger.info("\n" + "=" * 80)
        logger.info("QUARTERLY TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"MetaJudge model: {meta_judge_path}")
        logger.info(f"Edge library: {self.output_dir / 'edge_library.json'}")
        logger.info(f"Mistake library: {self.output_dir / 'mistake_library.json'}")
        logger.info("=" * 80)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _log_to_history(self, report: Dict, report_path: Path, meta_judge_path: Path):
        """Log training run to history."""
        pa = report['stages']['performance_analysis']
        pm = report['stages']['pattern_mining']
        mj = report['stages']['meta_judge_training']
        
        training_run = TrainingRun(
            training_id="",  # Will be auto-generated
            created_at="",  # Will be auto-generated
            symbol=self.symbol,
            period_from=self.start_date,
            period_to=self.end_date,
            total_trades=pa['total_trades'],
            good_trades=pa['good_trades'],
            bad_trades=pa['bad_trades'],
            edges_found=pm['edges_found'],
            mistakes_found=pm['mistakes_found'],
            meta_stats={
                "meta_auc": mj['auc'],
                "meta_accuracy": mj['accuracy'],
                "meta_precision": mj['precision'],
                "meta_recall": mj['recall']
            },
            paths={
                "report_path": str(report_path),
                "meta_model_path": str(meta_judge_path),
                "edge_library_path": str(self.output_dir / 'edge_library.json'),
                "mistake_library_path": str(self.output_dir / 'mistake_library.json')
            }
        )
        
        log_training_run(training_run)
        logger.info(f"Logged training run to history: {training_run.training_id}")
    
    def _print_summary(self, report: Dict):
        """Print training summary."""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        # Performance Analysis
        pa = report['stages']['performance_analysis']
        print(f"\n📊 Performance Analysis:")
        print(f"  Total Trades:  {pa['total_trades']}")
        print(f"  Good Trades:   {pa['good_trades']} ({100*pa['good_trades']/pa['total_trades']:.1f}%)")
        print(f"  Bad Trades:    {pa['bad_trades']} ({100*pa['bad_trades']/pa['total_trades']:.1f}%)")
        print(f"  Overall Winrate: {100*pa['winrate']:.1f}%")
        
        # Pattern Mining
        pm = report['stages']['pattern_mining']
        print(f"\n🔍 Pattern Mining:")
        print(f"  Edges Found:     {pm['edges_found']}")
        print(f"  Mistakes Found:  {pm['mistakes_found']}")
        if pm['top_edge']:
            print(f"  Top Edge:        {pm['top_edge']}")
        if pm['top_mistake']:
            print(f"  Top Mistake:     {pm['top_mistake']}")
        
        # MetaJudge
        mj = report['stages']['meta_judge_training']
        print(f"\n🧠 MetaJudge Training:")
        print(f"  AUC:       {mj['auc']:.4f}")
        print(f"  Accuracy:  {mj['accuracy']:.4f}")
        print(f"  Precision: {mj['precision']:.4f}")
        print(f"  Recall:    {mj['recall']:.4f}")
        
        print("\n" + "=" * 60)
        print("✅ The agent has learned from past trades and is now smarter!")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SCOPUS Quarterly Self-Learning Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on EURUSD Q1 2023
  python -m tools.run_quarterly_training \\
      --symbol EURUSD \\
      --from 2023-01-01 \\
      --to 2023-03-31 \\
      --output-dir models/quarterly_2023Q1

  # Train with custom thresholds
  python -m tools.run_quarterly_training \\
      --symbol GBPUSD \\
      --from 2023-04-01 \\
      --to 2023-06-30 \\
      --output-dir models/quarterly_2023Q2 \\
      --good-r 1.5 \\
      --min-edge-winrate 0.70
        """
    )
    
    parser.add_argument('--symbol', default='EURUSD', help="Trading symbol")
    parser.add_argument('--from', dest='start_date', required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument('--to', dest='end_date', required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument('--trades', default='backtest_results/trades.csv', help="Path to trades.csv")
    parser.add_argument('--output-dir', required=True, help="Output directory for models and reports")
    
    # PerformanceAnalyzer config
    parser.add_argument('--good-r', type=float, default=1.0, help="Min R-multiple for good trade")
    parser.add_argument('--bad-r', type=float, default=0.5, help="Max R-multiple for bad trade")
    
    # EdgeMiner config
    parser.add_argument('--min-support', type=int, default=5, help="Min trades for pattern")
    parser.add_argument('--min-edge-winrate', type=float, default=0.65, help="Min winrate for edge")
    
    # MetaJudge config
    parser.add_argument('--model-type', default='lightgbm', choices=['lightgbm', 'xgboost'])
    
    # Optional fine-tuning
    parser.add_argument('--finetune-ml', action='store_true', help="Fine-tune ML Brain")
    parser.add_argument('--finetune-rl', action='store_true', help="Fine-tune RL Brain")
    
    args = parser.parse_args()
    
    # Create output directory first
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Build config
    config = {
        'symbol': args.symbol,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'trades_path': args.trades,
        'output_dir': args.output_dir,
        'analyzer_config': {
            'good_trade_min_r': args.good_r,
            'bad_trade_max_r': args.bad_r
        },
        'miner_config': {
            'min_support': args.min_support,
            'min_edge_winrate': args.min_edge_winrate
        },
        'meta_judge_config': {
            'model_type': args.model_type
        },
        'finetune_ml_brain': args.finetune_ml,
        'finetune_rl_brain': args.finetune_rl
    }
    
    # Run orchestrator
    orchestrator = QuarterlyTrainingOrchestrator(config)
    report = orchestrator.run()
    
    print(f"\n✅ Training complete! Check {args.output_dir} for outputs.")


if __name__ == "__main__":
    main()
