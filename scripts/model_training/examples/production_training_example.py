#!/usr/bin/env python3
"""
修正后的模型训练脚本
使用优化后的配置重新训练模型，保留时间特征的临床合理性
"""

import sys
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_corrected_config():
    """Load corrected configuration"""
    import yaml
    
    config_path = Path(__file__).parent / "config" / "environment_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    return config_data

def load_and_prepare_data():
    """Load and prepare data with corrected configuration"""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_corrected_config()
    features_config = config['features']
    
    # Load dataset
    dataset_path = Path(__file__).parent.parent.parent / "data" / "model_ready" / "model_ready_dataset.csv"
    df = pd.read_csv(dataset_path)
    
    logger.info(f"📊 数据集加载完成: {df.shape}")
    logger.info(f"目标变量分布: {df['TARGET'].value_counts().to_dict()}")
    
    # Get features to remove (data leakage features)
    features_to_remove = features_config['data_leakage_features']
    logger.info(f"🚫 移除数据泄露特征: {features_to_remove}")
    
    # Get production features
    production_features = features_config['production_features']
    logger.info(f"✅ 使用生产特征: {len(production_features)}个")
    
    # Remove data leakage features
    df_clean = df.drop(columns=features_to_remove, errors='ignore')
    
    # Select production features
    available_features = [f for f in production_features if f in df_clean.columns]
    missing_features = [f for f in production_features if f not in df_clean.columns]
    
    if missing_features:
        logger.warning(f"⚠️ 缺失特征: {missing_features}")
    
    logger.info(f"📋 可用特征: {len(available_features)}个")
    
    # Prepare features and target
    X = df_clean[available_features].copy()
    y = df_clean['TARGET'].copy()
    
    # Handle target inversion
    invert_target = config['model']['invert_target']
    if invert_target:
        y = 1 - y
        logger.info("🔄 目标变量已反转: 正类(1) = 未治疗患者")
    
    logger.info(f"目标变量分布 (反转后): {y.value_counts().to_dict()}")
    
    return X, y, available_features, config

def preprocess_features(X, categorical_features):
    """Preprocess features"""
    logger = logging.getLogger(__name__)
    
    X_processed = X.copy()
    
    # Handle categorical features
    label_encoders = {}
    for feature in categorical_features:
        if feature in X_processed.columns:
            le = LabelEncoder()
            X_processed[feature] = le.fit_transform(X_processed[feature].astype(str))
            label_encoders[feature] = le
            logger.info(f"📊 编码分类特征: {feature}")
    
    # Handle missing values
    missing_counts = X_processed.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"🔧 处理缺失值: {missing_counts[missing_counts > 0].to_dict()}")
        X_processed = X_processed.fillna(X_processed.median())
    
    return X_processed, label_encoders

def train_model(X_train, y_train, config):
    """Train XGBoost model with corrected configuration"""
    logger = logging.getLogger(__name__)
    
    # Get hyperparameters
    hyperparams = config['hyperparameters']['xgboost']
    
    # Calculate scale_pos_weight for class imbalance
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1.0
    hyperparams['scale_pos_weight'] = scale_pos_weight
    
    logger.info(f"⚙️ XGBoost超参数: {hyperparams}")
    logger.info(f"📊 类别权重: {scale_pos_weight:.2f}")
    
    # Train model
    model = xgb.XGBClassifier(**hyperparams)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model performance"""
    logger = logging.getLogger(__name__)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info("📈 模型性能评估:")
    logger.info(f"  ROC-AUC: {roc_auc:.2f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"  Precision: {report['1']['precision']:.2f}")
    logger.info(f"  Recall: {report['1']['recall']:.2f}")
    logger.info(f"  F1-Score: {report['1']['f1-score']:.2f}")
    logger.info(f"  Accuracy: {report['accuracy']:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("🔍 特征重要性 (Top 10):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        logger.info(f"  {i:2d}. {row['feature']:<30} {row['importance']:.2f}")
    
    return {
        'roc_auc': roc_auc,
        'classification_report': report,
        'feature_importance': feature_importance,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def main():
    """Main training function"""
    logger = setup_logging()
    
    logger.info("🚀 开始修正后的模型训练")
    logger.info("=" * 80)
    
    try:
        # Load and prepare data
        X, y, feature_names, config = load_and_prepare_data()
        
        # Get categorical features
        categorical_features = config['features']['categorical_features']
        
        # Preprocess features
        X_processed, label_encoders = preprocess_features(X, categorical_features)
        
        # Split data
        test_size = config['model']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"📊 数据分割: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        
        # Apply SMOTE if configured
        use_smote = config['model']['use_smote']
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            logger.info(f"🔄 SMOTE应用: {X_train.shape} -> {X_train_smote.shape}")
            X_train, y_train = X_train_smote, y_train_smote
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = train_model(X_train_scaled, y_train, config)
        
        # Evaluate model
        results = evaluate_model(model, X_test_scaled, y_test, feature_names)
        
        # Save model and results
        model_dir = Path(__file__).parent.parent.parent / "backend" / "ml_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "xgboost_corrected_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = model_dir / "scaler_corrected.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save label encoders
        encoders_path = model_dir / "label_encoders_corrected.pkl"
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        
        # Save results
        results_path = model_dir / "training_results_corrected.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("💾 模型和结果已保存")
        logger.info(f"  模型: {model_path}")
        logger.info(f"  缩放器: {scaler_path}")
        logger.info(f"  编码器: {encoders_path}")
        logger.info(f"  结果: {results_path}")
        
        logger.info("🎉 修正后的模型训练完成!")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
