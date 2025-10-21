#!/usr/bin/env python3
"""
Pfizer EMR Alert System - Model Training Module
Unified entry point for all model training operations

Usage:
    python -m scripts.model_training train --environment production
    python -m scripts.model_training evaluate --environment production
    python -m scripts.model_training select --environment production
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def main():
    """Main entry point for model training module"""
    parser = argparse.ArgumentParser(
        description='Pfizer EMR Alert System - Model Training Module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python -m scripts.model_training train --environment production --model-type xgboost
  
  # Evaluate multiple models
  python -m scripts.model_training evaluate --environment production
  
  # Run model selection
  python -m scripts.model_training select --environment production
  
  # Run scheduled model selection
  python -m scripts.model_training select --scheduled --frequency 24
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--environment', type=str, default='production',
                             choices=['development', 'testing', 'staging', 'production'],
                             help='Environment configuration to use')
    train_parser.add_argument('--model-type', type=str, default=None,
                             choices=['xgboost', 'random_forest', 'logistic_regression', 
                                     'gradient_boosting', 'svm', 'naive_bayes'],
                             help='Model type to train')
    train_parser.add_argument('--dataset', type=str,
                             help='Path to dataset file')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--environment', type=str, default='production',
                            choices=['development', 'testing', 'staging', 'production'],
                            help='Environment configuration to use')
    eval_parser.add_argument('--models', nargs='+',
                            default=['xgboost', 'random_forest', 'logistic_regression', 
                                    'gradient_boosting', 'svm', 'naive_bayes'],
                            help='List of models to evaluate')
    eval_parser.add_argument('--dataset', type=str,
                            help='Path to dataset file')
    
    # Select command
    select_parser = subparsers.add_parser('select', help='Run model selection')
    select_parser.add_argument('--environment', type=str, default='production',
                              choices=['development', 'testing', 'staging', 'production'],
                              help='Environment to evaluate and update')
    select_parser.add_argument('--scheduled', action='store_true',
                              help='Run in scheduled mode with frequency checking')
    select_parser.add_argument('--frequency', type=int, default=24,
                              help='Hours between runs (for scheduled mode)')
    select_parser.add_argument('--force', action='store_true',
                              help='Force run regardless of schedule')
    select_parser.add_argument('--dry-run', action='store_true',
                              help='Evaluate models but do not update configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'train':
            from scripts.model_training.train import train_with_config
            from scripts.model_training.config.config_manager import Environment
            
            environment = Environment(args.environment)
            results = train_with_config(
                environment=environment,
                model_type=args.model_type,
                dataset_path=args.dataset
            )
            
            if results:
                print("✅ Training completed successfully!")
                return 0
            else:
                print("❌ Training failed!")
                return 1
                
        elif args.command == 'evaluate':
            from scripts.model_training.evaluate import evaluate_with_config
            from scripts.model_training.config.config_manager import Environment
            
            environment = Environment(args.environment)
            results = evaluate_with_config(
                environment=environment,
                model_types=args.models,
                dataset_path=args.dataset
            )
            
            if results:
                print("✅ Evaluation completed successfully!")
                return 0
            else:
                print("❌ Evaluation failed!")
                return 1
                
        elif args.command == 'select':
            if args.scheduled:
                # Run scheduled model selection
                from scripts.model_training.select_scheduled import main as scheduled_main
                
                # Set up arguments for scheduled script
                sys.argv = [
                    'select_scheduled.py',
                    '--environment', args.environment,
                    '--frequency', str(args.frequency)
                ]
                if args.force:
                    sys.argv.append('--force')
                if args.dry_run:
                    sys.argv.append('--dry-run')
                
                return scheduled_main()
            else:
                # Run immediate model selection
                from scripts.model_training.model_selection import main as selection_main
                return selection_main()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
