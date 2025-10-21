
Overfitting Analysis Report

Pfizer EMR Alert System – Model Overfitting Risk Assessment

⸻

🚨 Key Finding: Severe Data Leakage Detected

1. Evidence of Data Leakage

Main Problematic Feature:
	•	PHYS_TREAT_RATE_ALL (Physician historical treatment rate)
	•	Correlation with target variable: -0.606 (very high)
	•	Feature importance: 0.531 (dominant)
	•	Issue: This feature contains the physician’s historical treatment rate, which may include future information or be directly related to the target variable.

Other Suspicious Features:
	•	DIAGNOSIS_WITHIN_5DAYS_FLAG (Diagnosis within 5 days flag)
	•	Correlation: 0.239
	•	Feature importance: 0.081
	•	Issue: May directly reflect treatment time window; logically related to the target variable.
	•	SYMPTOM_TO_DIAGNOSIS_DAYS (Days from symptom onset to diagnosis)
	•	Correlation: 0.161
	•	Feature importance: 0.079
	•	Issue: Time-related features may affect treatment decisions.

2. Performance Difference Analysis

Baseline Model Results:
	•	Simplified feature set (suspicious features removed): AUC = 0.636
	•	Full feature set: AUC = 0.985
	•	Performance gap: 0.349 (large discrepancy indicates data leakage)

3. Overfitting Risk Assessment

✅ No evidence of traditional overfitting:
	•	Test vs. cross-validation performance difference is small (< 0.015)
	•	Cross-validation stability is high (coefficient of variation < 2%)
	•	Model generalization appears normal

❌ False high performance due to data leakage:
	•	Single feature (PHYS_TREAT_RATE_ALL) contributes 53% of total importance
	•	Removing suspicious features drastically reduces performance
	•	Model relies heavily on features that may contain future information

4. Data Quality Issues

Class imbalance:
	•	Positive samples: 92.2%
	•	Negative samples: 7.8%
	•	Imbalance ratio: 11.8:1 (severe imbalance)

Feature distribution:
	•	PATIENT_ID: 100% unique (normal)
	•	PHYSICIAN_ID: 81.5% unique (may cause overfitting)
	•	High number of missing values: 3,160

5. Recommended Solutions

🔧 Immediate Actions:
	1.	Rebuild the feature set
	•	Remove PHYS_TREAT_RATE_ALL (data leakage)
	•	Remove DIAGNOSIS_WITHIN_5DAYS_FLAG (logical leakage)
	•	Remove SYMPTOM_TO_DIAGNOSIS_DAYS (temporal leakage)
	2.	Use a more conservative feature set

Recommended features:
- PATIENT_AGE
- PATIENT_GENDER
- RISK_IMMUNO, RISK_CVD, RISK_DIABETES, RISK_OBESITY
- RISK_NUM, RISK_AGE_FLAG
- PHYS_EXPERIENCE_LEVEL (categorical)
- PHYSICIAN_STATE, PHYSICIAN_TYPE
- SYM_COUNT_5D
- DX_SEASON, LOCATION_TYPE, INSURANCE_TYPE_AT_DX
- PRIOR_CONTRA_LVL


	3.	Address class imbalance
	•	Apply SMOTE or ADASYN oversampling
	•	Adjust classification threshold
	•	Use stratified sampling

📊 Reevaluate the Model:
	1.	Retrain the model using the cleaned feature set
	2.	Expected AUC: 0.6–0.8 range
	3.	This will reflect the model’s true performance

🎯 Business Impact:
	•	The currently reported 98%+ accuracy is unrealistic
	•	True model performance is likely in the 60–80% AUC range
	•	Business expectations should be adjusted accordingly

6. Conclusion

The model exhibits severe data leakage, rather than traditional overfitting.

The extremely high performance (98%+ AUC) is mainly due to:
	1.	Inclusion of features containing future information
	2.	Features logically correlated with the target variable
	3.	Overreliance on a single dominant feature

Recommendation: Immediately rebuild the model with a conservative and logically sound feature set to obtain a realistic and reliable performance evaluation.

⸻

Analysis based on: 3,612 samples, 17 features, and 6 machine learning models


