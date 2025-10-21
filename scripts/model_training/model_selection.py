#!/usr/bin/env python3
"""
Dynamic Model Selection System
Automatically selects and updates the best performing model for production
"""

import sys
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.model_training.config.config_manager import ConfigManager, Environment
from scripts.model_training.pipelines.evaluation_pipeline import EvaluationPipeline


class DynamicModelSelector:
    """Dynamic model selection system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize dynamic model selector
        
        Args:
            project_root: Project root directory
        """
        if project_root is None:
            # Go up 3 levels from this file to reach project root
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = project_root
            
        self.config_manager = ConfigManager(self.project_root)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance thresholds
        self.min_performance_thresholds = {
            'pr_auc': 0.7,
            'roc_auc': 0.6,
            'recall': 0.8,
            'precision': 0.7
        }
        
        # Model selection criteria (priority order)
        self.selection_criteria = ['pr_auc', 'recall', 'precision', 'f1_score']
    
    def evaluate_all_models(self, environment: Environment = Environment.PRODUCTION) -> Dict[str, Any]:
        """
        Evaluate all available models
        
        Args:
            environment: Environment to use for evaluation
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info("🔍 Starting comprehensive model evaluation...")
        
        # Get configuration
        config = self.config_manager.get_config(environment)
        
        # Create evaluation pipeline
        pipeline = EvaluationPipeline(config)
        
        # Run evaluation
        model_types = ['xgboost', 'random_forest', 'logistic_regression', 
                      'gradient_boosting', 'svm', 'naive_bayes']
        
        results = pipeline.run_complete_evaluation(model_types=model_types)
        
        self.logger.info("✅ Model evaluation completed")
        return results
    
    def select_best_model(self, evaluation_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Select the best model based on performance criteria
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Best model information or None if no suitable model found
        """
        # Check for comparison results in different possible locations
        comparison_df = None
        if 'comparison_df' in evaluation_results:
            comparison_df = evaluation_results['comparison_df']
        elif 'best_model' in evaluation_results and evaluation_results['best_model']:
            # If we have best_model directly, return it
            return evaluation_results['best_model']
        else:
            self.logger.error("No comparison results found")
            return None
        
        if comparison_df is None:
            self.logger.error("No comparison results found")
            return None
        
        if len(comparison_df) == 0:
            self.logger.error("No models evaluated")
            return None
        
        # Filter models that meet minimum thresholds
        suitable_models = []
        for _, model in comparison_df.iterrows():
            meets_thresholds = True
            for metric, threshold in self.min_performance_thresholds.items():
                metric_key = metric.upper().replace('_', '-')
                if metric_key in model and model[metric_key] < threshold:
                    meets_thresholds = False
                    break
            
            if meets_thresholds:
                suitable_models.append(model)
        
        if not suitable_models:
            self.logger.warning("No models meet minimum performance thresholds")
            return None
        
        # Select best model based on criteria priority
        best_model = None
        for criterion in self.selection_criteria:
            criterion_key = criterion.upper().replace('_', '-')
            if criterion_key in comparison_df.columns:
                best_model = max(suitable_models, key=lambda x: x[criterion_key])
                break
        
        if best_model is None:
            self.logger.error("Could not select best model")
            return None
        
        self.logger.info(f"🏆 Selected best model: {best_model['Model']}")
        self.logger.info(f"📊 PR-AUC: {best_model['PR-AUC']:.2f}")
        self.logger.info(f"📈 Recall: {best_model['Recall']:.2f}")
        self.logger.info(f"🎯 Precision: {best_model['Precision']:.2f}")
        
        return best_model.to_dict()
    
    def update_production_config(self, best_model: Dict[str, Any], 
                               environment: Environment = Environment.PRODUCTION) -> bool:
        """
        Update production configuration with best model
        
        Args:
            best_model: Best model information
            environment: Environment to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            config_path = self.project_root / "scripts" / "model_training" / "config" / "environment_config.yaml"
            
            # Load current configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Update environment-specific configuration
            if 'environments' not in config_data:
                config_data['environments'] = {}
            
            if environment.value not in config_data['environments']:
                config_data['environments'][environment.value] = {}
            
            # Update model type
            config_data['environments'][environment.value]['model_type'] = best_model['Model'].lower().replace(' ', '_')
            
            # Save updated configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            self.logger.info(f"✅ Updated {environment.value} configuration with {best_model['Model']}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update configuration: {e}")
            return False
    
    def save_selection_log(self, best_model: Dict[str, Any], 
                          evaluation_results: Dict[str, Any]) -> bool:
        """
        Save model selection log
        
        Args:
            best_model: Best model information
            evaluation_results: Full evaluation results
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            log_dir = self.project_root / "logs" / "model_selection"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"model_selection_{timestamp}.json"
            
            log_data = {
                'timestamp': timestamp,
                'selected_model': best_model,
                'evaluation_summary': {
                    'total_models': len(evaluation_results.get('comparison_df', [])),
                    'best_pr_auc': best_model['PR-AUC'],
                    'best_recall': best_model['Recall'],
                    'best_precision': best_model['Precision'],
                    'best_f1_score': best_model['F1-Score']
                },
                'selection_criteria': self.selection_criteria,
                'performance_thresholds': self.min_performance_thresholds
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📝 Saved selection log: {log_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save selection log: {e}")
            return False
    
    def run_dynamic_selection(self, environment: Environment = Environment.PRODUCTION,
                            update_config: bool = True) -> Dict[str, Any]:
        """
        Run complete dynamic model selection process
        
        Args:
            environment: Environment to evaluate and update
            update_config: Whether to update configuration file
            
        Returns:
            Selection results dictionary
        """
        self.logger.info("🚀 Starting dynamic model selection process")
        self.logger.info("=" * 80)
        
        results = {
            'success': False,
            'selected_model': None,
            'evaluation_results': None,
            'config_updated': False,
            'log_saved': False,
            'error': None
        }
        
        try:
            # Step 1: Evaluate all models
            evaluation_results = self.evaluate_all_models(environment)
            results['evaluation_results'] = evaluation_results
            
            # Step 2: Select best model
            best_model = self.select_best_model(evaluation_results)
            if best_model is None:
                results['error'] = "No suitable model found"
                return results
            
            results['selected_model'] = best_model
            
            # Step 3: Update configuration if requested
            if update_config:
                config_updated = self.update_production_config(best_model, environment)
                results['config_updated'] = config_updated
                
                if not config_updated:
                    results['error'] = "Failed to update configuration"
                    return results
            
            # Step 4: Save selection log
            log_saved = self.save_selection_log(best_model, evaluation_results)
            results['log_saved'] = log_saved
            
            results['success'] = True
            self.logger.info("🎉 Dynamic model selection completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic model selection failed: {e}")
            results['error'] = str(e)
        
        return results


def main():
    """Main function for dynamic model selection"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create dynamic model selector
    selector = DynamicModelSelector()
    
    # Run dynamic selection
    results = selector.run_dynamic_selection(
        environment=Environment.PRODUCTION,
        update_config=True
    )
    
    if results['success']:
        print("\n✅ Dynamic model selection completed successfully!")
        print(f"🏆 Selected model: {results['selected_model']['Model']}")
        print(f"📊 PR-AUC: {results['selected_model']['PR-AUC']:.2f}")
        print(f"📈 Recall: {results['selected_model']['Recall']:.2f}")
        print(f"🎯 Precision: {results['selected_model']['Precision']:.2f}")
        print(f"⚖️ F1-Score: {results['selected_model']['F1-Score']:.2f}")
        
        if results['config_updated']:
            print("✅ Production configuration updated")
        
        if results['log_saved']:
            print("✅ Selection log saved")
    else:
        print(f"\n❌ Dynamic model selection failed: {results['error']}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
