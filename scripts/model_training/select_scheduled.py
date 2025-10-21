#!/usr/bin/env python3
"""
Scheduled Dynamic Model Selection
Runs automatic model selection on a schedule (e.g., daily, weekly)
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.model_training.model_selection import DynamicModelSelector
from scripts.model_training.config.config_manager import Environment


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(__file__).parent.parent.parent / "logs" / "dynamic_model_selection.log"
            )
        ]
    )


def should_run_selection(last_run_file: Path, frequency_hours: int = 24) -> bool:
    """
    Check if model selection should run based on frequency
    
    Args:
        last_run_file: Path to file storing last run timestamp
        frequency_hours: Hours between runs
        
    Returns:
        True if selection should run, False otherwise
    """
    if not last_run_file.exists():
        return True
    
    try:
        with open(last_run_file, 'r') as f:
            last_run_str = f.read().strip()
        
        last_run = datetime.fromisoformat(last_run_str)
        next_run = last_run + timedelta(hours=frequency_hours)
        
        return datetime.now() >= next_run
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Error reading last run file: {e}")
        return True


def update_last_run_file(last_run_file: Path):
    """Update the last run timestamp file"""
    try:
        last_run_file.parent.mkdir(parents=True, exist_ok=True)
        with open(last_run_file, 'w') as f:
            f.write(datetime.now().isoformat())
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to update last run file: {e}")


def main():
    """Main function for scheduled model selection"""
    parser = argparse.ArgumentParser(
        description='Scheduled Dynamic Model Selection for Pfizer EMR Alert System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run immediately (ignore schedule)
  python3 scripts/model_training/scheduled_model_selector.py --force
  
  # Run with custom frequency (every 12 hours)
  python3 scripts/model_training/scheduled_model_selector.py --frequency 12
  
  # Run in dry-run mode (evaluate but don't update config)
  python3 scripts/model_training/scheduled_model_selector.py --dry-run
  
  # Run with custom environment
  python3 scripts/model_training/scheduled_model_selector.py --environment staging
        """
    )
    
    parser.add_argument('--force', action='store_true',
                       help='Force run regardless of schedule')
    parser.add_argument('--frequency', type=int, default=24,
                       help='Hours between runs (default: 24)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Evaluate models but do not update configuration')
    parser.add_argument('--environment', type=str, default='production',
                       choices=['development', 'testing', 'staging', 'production'],
                       help='Environment to evaluate and update')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Convert environment string to enum
    try:
        environment = Environment(args.environment)
    except ValueError:
        logger.error(f"Invalid environment: {args.environment}")
        return 1
    
    # Check if selection should run
    project_root = Path(__file__).parent.parent.parent
    last_run_file = project_root / "logs" / "last_model_selection_run.txt"
    
    if not args.force and not should_run_selection(last_run_file, args.frequency):
        logger.info("⏰ Model selection not due yet, skipping...")
        return 0
    
    logger.info("🚀 Starting scheduled model selection")
    logger.info(f"Environment: {environment.value}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Frequency: {args.frequency} hours")
    
    try:
        # Create dynamic model selector
        selector = DynamicModelSelector(project_root)
        
        # Run dynamic selection
        results = selector.run_dynamic_selection(
            environment=environment,
            update_config=not args.dry_run
        )
        
        if results['success']:
            logger.info("✅ Scheduled model selection completed successfully!")
            
            # Update last run file
            update_last_run_file(last_run_file)
            
            # Log results
            selected_model = results['selected_model']
            logger.info(f"🏆 Selected model: {selected_model['Model']}")
            logger.info(f"📊 PR-AUC: {selected_model['PR-AUC']:.2f}")
            logger.info(f"📈 Recall: {selected_model['Recall']:.2f}")
            logger.info(f"🎯 Precision: {selected_model['Precision']:.2f}")
            logger.info(f"⚖️ F1-Score: {selected_model['F1-Score']:.2f}")
            
            if args.dry_run:
                logger.info("🔍 Dry run completed - no configuration updated")
            else:
                logger.info("✅ Production configuration updated")
            
            return 0
        else:
            logger.error(f"❌ Scheduled model selection failed: {results['error']}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Unexpected error during scheduled model selection: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
