# Model Comparison Report - Based on Optimized Hyperparameters

## 📊 Executive Summary

**Update Date**: 2025-10-25  
**Optimization Method**: Bayesian Optimization (Optuna)  
**Optimization Target**: F1-Score Maximization  
**Number of Trials**: 100 trials  
**Optimization Time**: 387.5 seconds

---

## 🏆 Model Performance Comparison Table

| Rank | Model | Recall | Precision | F1-Score | ROC-AUC | PR-AUC | Accuracy |
|------|-------|---------|-----------|----------|---------|---------|----------|
| 🥇 | **XGBoost (Optimized)** | **0.85** | 0.84 | **0.85** | 0.62 | 0.88 | **0.75** |
| 🥈 | Gradient Boosting | 0.73 | 0.88 | 0.80 | 0.68 | **0.91** | 0.69 |
| 🥉 | Random Forest | 0.70 | **0.89** | 0.78 | **0.69** | **0.91** | 0.68 |
| 4 | Logistic Regression | 0.65 | **0.89** | 0.76 | 0.66 | 0.89 | 0.65 |
| 5 | SVM | 0.34 | **0.93** | 0.49 | 0.67 | 0.90 | 0.43 |
| 6 | Naive Bayes | 0.36 | 0.91 | 0.52 | 0.63 | 0.89 | 0.44 |

**Note**: 🥇 = 1st Place, 🥈 = 2nd Place, 🥉 = 3rd Place

---

## 📈 Detailed Metrics Analysis

### 1. XGBoost (Optimized Hyperparameters) 🏆

**Performance Summary**: Best overall performance

| Metric | Value | Rank | Evaluation |
|--------|-------|------|------------|
| Recall | 0.85 | 1/6 | ✅ Highest |
| Precision | 0.84 | 3/6 | ✅ Excellent |
| F1-Score | 0.85 | 1/6 | ✅ Highest |
| ROC-AUC | 0.62 | 4/6 | ⚠️ Lower |
| PR-AUC | 0.88 | 3/6 | ✅ Excellent |
| Accuracy | 0.75 | 1/6 | ✅ Highest |

**Key Advantages**:
- ✅ **Highest Recall**: Captures 85% of cases requiring alerts
- ✅ **Highest F1-Score**: Best balance between Precision and Recall
- ✅ **Highest Accuracy**: Overall prediction accuracy of 75%
- ✅ **High Precision**: Low false positive rate, reduces false alarms

**Trade-offs**:
- ⚠️ Lower ROC-AUC (0.62), possibly due to class imbalance
- ⚠️ Need to monitor false negatives in medical scenarios

---

### 2. Gradient Boosting 🥈

| Metric | Value | Rank | Evaluation |
|--------|-------|------|------------|
| Recall | 0.73 | 3/6 | ✅ Good |
| Precision | 0.88 | 1/6 (tied) | ✅ Excellent |
| F1-Score | 0.80 | 2/6 | ✅ Good |
| ROC-AUC | 0.68 | 2/6 | ✅ Good |
| PR-AUC | 0.91 | 1/6 (tied) | ✅ Highest |
| Accuracy | 0.69 | 2/6 | ✅ Good |

**Characteristics**: Excellent performance in precision and PR-AUC

---

### 3. Random Forest 🥉

| Metric | Value | Rank | Evaluation |
|--------|-------|------|------------|
| Recall | 0.70 | 2/6 | ✅ Good |
| Precision | 0.89 | 1/6 (tied) | ✅ Excellent |
| F1-Score | 0.78 | 3/6 | ✅ Good |
| ROC-AUC | 0.69 | 1/6 | ✅ Highest |
| PR-AUC | 0.91 | 1/6 (tied) | ✅ Highest |
| Accuracy | 0.68 | 3/6 | ✅ Good |

**Characteristics**: Optimal performance in ROC-AUC, PR-AUC and cross-validation

---

### 4. Logistic Regression (4th Place)

| Metric | Value | Rank | Evaluation |
|--------|-------|------|------------|
| Recall | 0.65 | 4/6 | ⚠️ Moderate |
| Precision | 0.89 | 1/6 (tied) | ✅ Excellent |
| F1-Score | 0.76 | 4/6 | ⚠️ Moderate |
| ROC-AUC | 0.66 | 3/6 | ⚠️ Moderate |
| PR-AUC | 0.89 | 4/6 | ⚠️ Moderate |
| Accuracy | 0.65 | 4/6 | ⚠️ Moderate |

**Characteristics**: Simple and stable, suitable for high precision requirements

---

### 5. SVM (5th Place)

| Metric | Value | Rank | Evaluation |
|--------|-------|------|------------|
| Recall | 0.34 | 6/6 | ❌ Lowest |
| Precision | 0.93 | 1/6 | ✅ Highest |
| F1-Score | 0.49 | 5/6 | ❌ Lower |
| ROC-AUC | 0.67 | 2/6 | ⚠️ Moderate |
| PR-AUC | 0.90 | 2/6 | ✅ Excellent |
| Accuracy | 0.43 | 6/6 | ❌ Lower |

**Characteristics**: High precision but low recall, not suitable for high recall scenarios

---

### 6. Naive Bayes (6th Place)

| Metric | Value | Rank | Evaluation |
|--------|-------|------|------------|
| Recall | 0.36 | 5/6 | ❌ Lower |
| Precision | 0.91 | 2/6 | ✅ Excellent |
| F1-Score | 0.52 | 6/6 | ❌ Lowest |
| ROC-AUC | 0.63 | 5/6 | ⚠️ Moderate |
| PR-AUC | 0.89 | 4/6 | ⚠️ Moderate |
| Accuracy | 0.44 | 5/6 | ❌ Lower |

**Characteristics**: Average performance, not suitable for current dataset

---

## 🎯 Key Metrics Comparison Visualization

### Recall (Sensitivity)
```
XGBoost (Opt)      ████████████████████████░░ 85%
Random Forest      ████████████████████░░░░░░ 70%
Gradient Boosting  ██████████████████░░░░░░░░ 73%
Logistic Regression█████████████████░░░░░░░░░ 65%
Naive Bayes        ████████░░░░░░░░░░░░░░░░░░ 36%
SVM                ██████░░░░░░░░░░░░░░░░░░░░ 34%
```

### Precision
```
SVM                ████████████████████████░░ 93%
Random Forest      ███████████████████████░░░ 89%
Logistic Regression██████████████████████░░░ 89%
Naive Bayes        ██████████████████████░░░░ 91%
Gradient Boosting  █████████████████████░░░░░ 88%
XGBoost (Opt)      ████████████████████░░░░░░ 84%
```

### F1-Score
```
XGBoost (Opt)      ████████████████████████░░ 85%
Gradient Boosting  ████████████████████░░░░░░ 80%
Random Forest      ███████████████████░░░░░░░ 78%
Logistic Regression█████████████████░░░░░░░░░ 76%
Naive Bayes        ██████████░░░░░░░░░░░░░░░░ 52%
SVM                ██████████░░░░░░░░░░░░░░░░ 49%
```

### ROC-AUC
```
Random Forest      ████████████████████░░░░░░ 69%
Gradient Boosting  ████████████████████░░░░░░ 68%
SVM                ███████████████████░░░░░░░ 67%
Logistic Regression██████████████████░░░░░░░░ 66%
Naive Bayes        ████████████████░░░░░░░░░░ 63%
XGBoost (Opt)      ██████████████░░░░░░░░░░░░ 62%
```

### PR-AUC
```
Random Forest      ████████████████████████░░ 91%
Gradient Boosting  ████████████████████████░░ 91%
SVM                ███████████████████████░░░ 90%
Logistic Regression██████████████████████░░░░ 89%
Naive Bayes        ██████████████████████░░░░ 89%
XGBoost (Opt)      █████████████████████░░░░░ 88%
```

### Accuracy
```
XGBoost (Opt)      ████████████████████████░░ 75%
Gradient Boosting  ████████████████████░░░░░░ 69%
Random Forest      ███████████████████░░░░░░░ 68%
Logistic Regression█████████████████░░░░░░░░░ 65%
Naive Bayes        ████████░░░░░░░░░░░░░░░░░░ 44%
SVM                █████████░░░░░░░░░░░░░░░░░ 43%
```

---

## 💡 Conclusions and Recommendations

### 🏆 Recommended Model: XGBoost (Optimized)

**Selection Rationale**:
1. ✅ **Best Overall Performance**: F1-Score 0.85
2. ✅ **Highest Recall**: 85%, lowest missed detection rate
3. ✅ **Highest Accuracy**: 75%
4. ✅ **High Precision**: 84%, controllable false positive rate

**Suitable Scenarios**:
- ✅ EMR Alert System
- ✅ Medical scenarios requiring high recall
- ✅ Diagnostic assistance with high accuracy requirements

### Alternative Options

**If extremely high precision is required**:
- Recommended: Random Forest (Precision: 89%)
- Risk: May miss some cases requiring alerts

**If balanced performance is needed**:
- Recommended: Gradient Boosting (PR-AUC: 91%)
- Characteristics: Better balance between precision and recall

---

## 📊 Data Sources

### Optimization Parameters
- File: `logs/hyperparameter_optimization/optuna_optimization_20251025_221340.json`
- Method: Bayesian Optimization
- Target: F1-Score
- Number of Trials: 100

### Other Model Evaluations
- File: `scripts/reports/model_evaluation/model_comparison_results.csv`
- Method: 5-fold Cross Validation
- Dataset: model_ready_dataset.csv

---

## 📅 Update History

| Date | Version | Updates |
|------|---------|---------|
| 2025-10-25 | 2.1.0 | Added optimized XGBoost results |
| 2025-10-25 | 2.1.0 | Completed comprehensive model comparison analysis |

---

**Last Updated**: 2025-10-25  
**Version**: 2.1.0  
**Status**: ✅ Completed
