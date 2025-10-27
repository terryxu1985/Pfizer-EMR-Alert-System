#!/usr/bin/env python3
"""
Create XGBoost-specific Feature Importance Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def create_xgboost_importance_visualization():
    """Create detailed XGBoost feature importance visualization"""
    
    # Read the feature importance data
    csv_path = Path(__file__).parent.parent.parent / "reports/model_evaluation/feature_importance/detailed_feature_importance_analysis.csv"
    df = pd.read_csv(csv_path)
    
    # Sort by XGBoost importance (descending)
    df_sorted = df.sort_values('XGBoost_Importance', ascending=False)
    
    # Filter out zero-importance features for cleaner visualization
    df_filtered = df_sorted[df_sorted['XGBoost_Importance'] > 0]
    
    # Create figure with subplots (larger to accommodate 32 features)
    fig = plt.figure(figsize=(16, 18))
    
    # Plot 1: Top 20 Features - Bar Chart
    ax1 = plt.subplot(2, 2, 1)
    top_20 = df_filtered.head(20)
    ax1.barh(range(len(top_20)), top_20['XGBoost_Importance'].values, color='#006400')
    ax1.set_yticks(range(len(top_20)))
    ax1.set_yticklabels(top_20['Feature'].values, fontsize=9)
    ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Features - XGBoost Model', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_20.iterrows()):
        ax1.text(row['XGBoost_Importance'] + 0.001, i, f'{row["XGBoost_Importance"]:.3f}', 
                va='center', fontsize=8)
    
    # Plot 2: Feature Importance Distribution
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(df_filtered['XGBoost_Importance'].values, bins=15, color='#006400', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Feature Importance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: All Features - Detailed View
    ax3 = plt.subplot(2, 2, 3)
    all_features = df_filtered  # Show all features with non-zero importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_features)))
    bars = ax3.barh(range(len(all_features)), all_features['XGBoost_Importance'].values, color=colors)
    ax3.set_yticks(range(len(all_features)))
    ax3.set_yticklabels(all_features['Feature'].values, fontsize=6)
    ax3.set_xlabel('Importance Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax3.set_title(f'All {len(all_features)} Features - Complete View', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars (only for top 5 features to avoid clutter)
    for i, (bar, (idx, row)) in enumerate(zip(bars, all_features.iterrows())):
        if i < 5:  # Only show labels for top 5 features to avoid clutter
            ax3.text(row['XGBoost_Importance'] + 0.001, i, f'{row["XGBoost_Importance"]:.3f}', 
                    va='center', fontsize=6)
    
    # Plot 4: Cumulative Importance
    ax4 = plt.subplot(2, 2, 4)
    df_cumulative = df_filtered.copy()
    df_cumulative['cumulative'] = df_cumulative['XGBoost_Importance'].cumsum()
    df_cumulative['percentage'] = (df_cumulative['XGBoost_Importance'] / df_cumulative['XGBoost_Importance'].sum() * 100).cumsum()
    
    # Plot cumulative percentage - show all features
    features_to_show = len(df_cumulative)
    ax4.plot(range(1, features_to_show + 1), df_cumulative['percentage'].head(features_to_show), 
             marker='o', linewidth=2, markersize=4, color='#006400')
    ax4.axhline(y=80, color='red', linestyle='--', linewidth=1.5, label='80% Threshold')
    ax4.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Importance (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Cumulative Feature Importance (All Features)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, min(35, features_to_show + 1))
    ax4.set_ylim(0, 100)
    
    # Find how many features make up 80% of importance
    features_for_80 = (df_cumulative['percentage'] <= 80).sum()
    ax4.axvline(x=features_for_80, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(__file__).parent.parent.parent / "reports/model_evaluation/feature_importance/xgboost_feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ XGBoost feature importance visualization saved to: {output_path}")
    
    # Create a detailed report
    report_path = Path(__file__).parent.parent.parent / "reports/model_evaluation/feature_importance/xgboost_feature_importance_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("XGBOOST MODEL - FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Features Analyzed: {len(df)}\n")
        f.write(f"Features with Non-Zero Importance: {len(df_filtered)}\n")
        f.write(f"Features with Zero Importance: {len(df) - len(df_filtered)}\n\n")
        
        f.write("TOP 20 MOST IMPORTANT FEATURES\n")
        f.write("-" * 40 + "\n")
        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            f.write(f"{idx:2d}. {row['Feature']:35s} : {row['XGBoost_Importance']:6.4f}\n")
        
        # Statistics
        f.write("\nFEATURE IMPORTANCE STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Importance: {df_filtered['XGBoost_Importance'].mean():.6f}\n")
        f.write(f"Median Importance: {df_filtered['XGBoost_Importance'].median():.6f}\n")
        f.write(f"Max Importance: {df_filtered['XGBoost_Importance'].max():.6f}\n")
        f.write(f"Min Importance: {df_filtered['XGBoost_Importance'].min():.6f}\n")
        f.write(f"Std Importance: {df_filtered['XGBoost_Importance'].std():.6f}\n")
        
        # Cumulative analysis
        f.write("\nCUMULATIVE IMPORTANCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        cumulative_sum = 0
        for idx, (_, row) in enumerate(df_cumulative.iterrows(), 1):
            cumulative_sum += row['XGBoost_Importance']
            if idx <= 5 or idx % 5 == 0 or cumulative_sum / df_cumulative['XGBoost_Importance'].sum() >= 0.8:
                pct = (cumulative_sum / df_cumulative['XGBoost_Importance'].sum()) * 100
                f.write(f"Top {idx:2d} features account for {pct:6.2f}% of total importance\n")
                if cumulative_sum / df_cumulative['XGBoost_Importance'].sum() >= 0.8 and idx <= 10:
                    break
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Insights
        top_5 = df_filtered.head(5)
        f.write("1. TOP 5 CRITICAL FEATURES:\n")
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            pct = (row['XGBoost_Importance'] / df_filtered['XGBoost_Importance'].sum()) * 100
            f.write(f"   - {row['Feature']} ({pct:.1f}% of total importance)\n")
        
        f.write("\n2. MEDICAL DOMAIN ANALYSIS:\n")
        # Categorize features
        demographic = ['PATIENT_AGE', 'PATIENT_GENDER', 'RISK_AGE_FLAG']
        clinical = ['RISK_NUM', 'RISK_CVD', 'RISK_IMMUNO', 'RISK_DIABETES', 'RISK_OBESITY', 'PRIOR_CONTRA_LVL']
        provider = ['PHYSICIAN_TYPE', 'PHYSICIAN_STATE', 'PHYS_TOTAL_DX', 'PHYS_EXPERIENCE_LEVEL']
        temporal = ['SYMPTOM_TO_DIAGNOSIS_DAYS', 'SYM_COUNT_5D', 'DIAGNOSIS_WITHIN_5DAYS_FLAG', 'DX_SEASON']
        location = ['LOCATION_TYPE']
        symptoms = [col for col in df_filtered['Feature'] if 'SYMPTOM' in col]
        
        def get_category_score(categories):
            return df_filtered[df_filtered['Feature'].isin(categories)]['XGBoost_Importance'].sum()
        
        f.write(f"   - Demographic Features: {get_category_score(demographic):.3f}\n")
        f.write(f"   - Clinical Risk Features: {get_category_score(clinical):.3f}\n")
        f.write(f"   - Provider Features: {get_category_score(provider):.3f}\n")
        f.write(f"   - Temporal Features: {get_category_score(temporal):.3f}\n")
        f.write(f"   - Location Features: {get_category_score(location):.3f}\n")
        f.write(f"   - Symptom Features: {get_category_score(symptoms):.3f}\n")
        
    print(f"✅ XGBoost feature importance report saved to: {report_path}")
    
    # Also create a CSV with just XGBoost data
    csv_output_path = Path(__file__).parent.parent.parent / "reports/model_evaluation/feature_importance/xgboost_only_importance.csv"
    xgboost_data = df[['Feature', 'XGBoost_Importance']].sort_values('XGBoost_Importance', ascending=False)
    xgboost_data.to_csv(csv_output_path, index=False)
    print(f"✅ XGBoost-only CSV saved to: {csv_output_path}")

if __name__ == "__main__":
    create_xgboost_importance_visualization()
