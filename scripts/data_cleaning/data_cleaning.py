"""
Data Cleaning Pipeline for Pfizer EMR Alert System

This script cleans three related datasets:
- fact_txn.xlsx: Transaction fact table
- dim_physician.xlsx: Physician dimension table
- dim_patient.xlsx: Patient dimension table

Implements comprehensive cleaning rules including:
- Format standardization
- Missing value handling
- Referential integrity validation
- Logical consistency checks
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Main data cleaning class with integrated quality assessment"""
    
    def __init__(self, raw_dir, cleaned_dir, generate_quality_report=True):
        """
        Initialize data cleaner
        
        Args:
            raw_dir: Directory containing raw data files
            cleaned_dir: Directory for cleaned output files
            generate_quality_report: Whether to generate quality assessment report
        """
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.generate_quality_report = generate_quality_report
        
        # Ensure output directory exists
        os.makedirs(cleaned_dir, exist_ok=True)
        
        # Store dataframes
        self.df_patient = None
        self.df_physician = None
        self.df_transaction = None
        
        # Store cleaning statistics
        self.cleaning_stats = {
            'patient': {},
            'physician': {},
            'transaction': {}
        }
        
        # Store quality assessment results
        self.quality_results = {
            'patient': {},
            'physician': {},
            'transaction': {}
        }
    
    def load_data(self):
        """Load all raw data files"""
        print("=" * 80)
        print("Loading Raw Data Files")
        print("=" * 80)
        
        try:
            # Load patient data
            patient_path = os.path.join(self.raw_dir, 'dim_patient.xlsx')
            self.df_patient = pd.read_excel(patient_path)
            print(f"✓ Loaded dim_patient: {self.df_patient.shape}")
            
            # Load physician data
            physician_path = os.path.join(self.raw_dir, 'dim_physician.xlsx')
            self.df_physician = pd.read_excel(physician_path)
            print(f"✓ Loaded dim_physician: {self.df_physician.shape}")
            
            # Load transaction data
            transaction_path = os.path.join(self.raw_dir, 'fact_txn.xlsx')
            self.df_transaction = pd.read_excel(transaction_path)
            print(f"✓ Loaded fact_txn: {self.df_transaction.shape}")
            
            return True
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return False
    
    def clean_patient_data(self):
        """Clean patient dimension table"""
        print("\n" + "=" * 80)
        print("Cleaning Patient Dimension Table (dim_patient)")
        print("=" * 80)
        
        df = self.df_patient.copy()
        initial_count = len(df)
        stats = {}
        
        # 1. Birth year validation
        print("\n1. Validating BIRTH_YEAR...")
        current_year = datetime.now().year
        invalid_birth_year = (df['BIRTH_YEAR'] < 1900) | (df['BIRTH_YEAR'] > current_year)
        stats['invalid_birth_year'] = invalid_birth_year.sum()
        print(f"   - Found {stats['invalid_birth_year']} invalid birth years")
        
        # Replace invalid birth years with -1 (unknown marker)
        if stats['invalid_birth_year'] > 0:
            df.loc[invalid_birth_year, 'BIRTH_YEAR'] = -1
            print(f"   - Set invalid birth years to -1 (Unknown)")
        
        # Fill missing birth years with -1 (unknown marker)
        missing_birth_year = df['BIRTH_YEAR'].isna()
        stats['missing_birth_year'] = missing_birth_year.sum()
        if stats['missing_birth_year'] > 0:
            df['BIRTH_YEAR'] = df['BIRTH_YEAR'].fillna(-1)
            print(f"   - Filled {stats['missing_birth_year']} missing birth years with -1 (Unknown)")
        
        # 2. Gender normalization
        print("\n2. Normalizing GENDER...")
        
        # Handle blank values first
        df['GENDER'] = df['GENDER'].astype(str).str.strip()
        df['GENDER'] = df['GENDER'].replace(['', 'nan', 'None', 'NaN'], 'Unknown')
        
        original_gender_counts = df['GENDER'].value_counts()
        print(f"   - Original values: {dict(original_gender_counts)}")
        
        # Standardize gender values
        gender_map = {
            'M': 'M', 'm': 'M', 'Male': 'M', 'male': 'M',
            'F': 'F', 'f': 'F', 'Female': 'F', 'female': 'F',
            'Unknown': 'Unknown'
        }
        df['GENDER'] = df['GENDER'].map(gender_map).fillna('Unknown')
        
        new_gender_counts = df['GENDER'].value_counts()
        print(f"   - Standardized values: {dict(new_gender_counts)}")
        
        # 3. Check for duplicates
        print("\n3. Checking for duplicates...")
        duplicates = df.duplicated(subset=['PATIENT_ID'], keep='first')
        stats['duplicates'] = duplicates.sum()
        print(f"   - Found {stats['duplicates']} duplicate PATIENT_ID records")
        
        if stats['duplicates'] > 0:
            df = df[~duplicates]
            print(f"   - Removed duplicates, keeping first occurrence")
        
        # 4. Ensure PATIENT_ID is valid
        print("\n4. Validating PATIENT_ID...")
        invalid_ids = (df['PATIENT_ID'] <= 0) | df['PATIENT_ID'].isna()
        stats['invalid_patient_id'] = invalid_ids.sum()
        print(f"   - Found {stats['invalid_patient_id']} invalid PATIENT_ID values")
        
        if stats['invalid_patient_id'] > 0:
            df = df[~invalid_ids]
            print(f"   - Removed invalid records")
        
        # 5. Sort by PATIENT_ID
        df = df.sort_values('PATIENT_ID').reset_index(drop=True)
        
        # Summary
        final_count = len(df)
        stats['records_removed'] = initial_count - final_count
        stats['final_count'] = final_count
        
        print("\n" + "-" * 80)
        print("Patient Data Cleaning Summary:")
        print(f"  Initial records: {initial_count:,}")
        print(f"  Final records: {final_count:,}")
        print(f"  Records removed: {stats['records_removed']:,}")
        print(f"  Invalid birth years fixed: {stats['invalid_birth_year']}")
        print(f"  Duplicates removed: {stats['duplicates']}")
        print("-" * 80)
        
        self.df_patient = df
        self.cleaning_stats['patient'] = stats
        
        return df
    
    def clean_physician_data(self):
        """Clean physician dimension table"""
        print("\n" + "=" * 80)
        print("Cleaning Physician Dimension Table (dim_physician)")
        print("=" * 80)
        
        df = self.df_physician.copy()
        initial_count = len(df)
        stats = {}
        
        # 1. Handle missing values
        print("\n1. Analyzing missing values...")
        missing_stats = df.isnull().sum()
        for col in missing_stats.index:
            pct = (missing_stats[col] / len(df) * 100)
            if missing_stats[col] > 0:
                print(f"   - {col}: {missing_stats[col]:,} ({pct:.2f}%)")
        
        # 2. Birth year validation
        print("\n2. Validating BIRTH_YEAR...")
        current_year = datetime.now().year
        valid_birth_mask = (df['BIRTH_YEAR'] >= 1930) & (df['BIRTH_YEAR'] <= 2005)
        invalid_birth_year = (~valid_birth_mask) & df['BIRTH_YEAR'].notna()
        stats['invalid_birth_year'] = invalid_birth_year.sum()
        print(f"   - Found {stats['invalid_birth_year']} invalid birth years")
        
        if stats['invalid_birth_year'] > 0:
            # Replace invalid with -1 (unknown marker)
            df.loc[invalid_birth_year, 'BIRTH_YEAR'] = -1
            print(f"   - Set invalid birth years to -1 (Unknown)")
        
        # Fill missing birth years with -1 (unknown marker)
        missing_birth_year = df['BIRTH_YEAR'].isna()
        stats['missing_birth_year'] = missing_birth_year.sum()
        if stats['missing_birth_year'] > 0:
            df['BIRTH_YEAR'] = df['BIRTH_YEAR'].fillna(-1)
            print(f"   - Filled {stats['missing_birth_year']} missing birth years with -1 (Unknown)")
        
        # 3. Gender normalization
        print("\n3. Normalizing GENDER...")
        # Handle blank values
        df['GENDER'] = df['GENDER'].astype(str).str.strip()
        df['GENDER'] = df['GENDER'].replace(['', 'nan', 'None', 'NaN'], 'Unknown')
        
        gender_map = {
            'M': 'M', 'm': 'M', 'Male': 'M', 'male': 'M',
            'F': 'F', 'f': 'F', 'Female': 'F', 'female': 'F',
            'Unknown': 'Unknown'
        }
        df['GENDER'] = df['GENDER'].map(gender_map).fillna('Unknown')
        print(f"   - Standardized values: {dict(df['GENDER'].value_counts())}")
        
        # 4. State field standardization
        print("\n4. Standardizing STATE...")
        if 'STATE' in df.columns:
            # Handle blank values
            df['STATE'] = df['STATE'].astype(str).str.strip()
            df['STATE'] = df['STATE'].replace(['', 'nan', 'None', 'NaN'], 'Unknown')
            df['STATE'] = df['STATE'].str.upper()
            print(f"   - Unique states: {df['STATE'].nunique()}")
        
        # 5. Physician type normalization
        print("\n5. Normalizing PHYSICIAN_TYPE...")
        if 'PHYSICIAN_TYPE' in df.columns:
            # Handle blank values
            df['PHYSICIAN_TYPE'] = df['PHYSICIAN_TYPE'].astype(str).str.strip()
            df['PHYSICIAN_TYPE'] = df['PHYSICIAN_TYPE'].replace(['', 'nan', 'None', 'NaN'], 'Unknown')
            df['PHYSICIAN_TYPE'] = df['PHYSICIAN_TYPE'].str.title()
            print(f"   - Unique types: {df['PHYSICIAN_TYPE'].nunique()}")
        
        # 6. Specialty standardization
        print("\n6. Standardizing SPECIALTY...")
        if 'SPECIALTY' in df.columns:
            # Handle blank values
            df['SPECIALTY'] = df['SPECIALTY'].astype(str).str.strip()
            df['SPECIALTY'] = df['SPECIALTY'].replace(['', 'nan', 'None', 'NaN'], 'Unknown')
            df['SPECIALTY'] = df['SPECIALTY'].str.upper()
            print(f"   - Unique specialties: {df['SPECIALTY'].nunique()}")
        
        # 7. Check for duplicates
        print("\n7. Checking for duplicates...")
        duplicates = df.duplicated(subset=['PHYSICIAN_ID'], keep='first')
        stats['duplicates'] = duplicates.sum()
        print(f"   - Found {stats['duplicates']} duplicate PHYSICIAN_ID records")
        
        if stats['duplicates'] > 0:
            df = df[~duplicates]
            print(f"   - Removed duplicates")
        
        # 8. Validate PHYSICIAN_ID
        print("\n8. Validating PHYSICIAN_ID...")
        invalid_ids = (df['PHYSICIAN_ID'] <= 0) | df['PHYSICIAN_ID'].isna()
        stats['invalid_physician_id'] = invalid_ids.sum()
        print(f"   - Found {stats['invalid_physician_id']} invalid PHYSICIAN_ID values")
        
        if stats['invalid_physician_id'] > 0:
            df = df[~invalid_ids]
            print(f"   - Removed invalid records")
        
        # 9. Sort by PHYSICIAN_ID
        df = df.sort_values('PHYSICIAN_ID').reset_index(drop=True)
        
        # Summary
        final_count = len(df)
        stats['records_removed'] = initial_count - final_count
        stats['final_count'] = final_count
        
        print("\n" + "-" * 80)
        print("Physician Data Cleaning Summary:")
        print(f"  Initial records: {initial_count:,}")
        print(f"  Final records: {final_count:,}")
        print(f"  Records removed: {stats['records_removed']:,}")
        print(f"  Invalid birth years: {stats['invalid_birth_year']}")
        print(f"  Duplicates removed: {stats['duplicates']}")
        print("-" * 80)
        
        self.df_physician = df
        self.cleaning_stats['physician'] = stats
        
        return df
    
    def clean_transaction_data(self):
        """Clean transaction fact table"""
        print("\n" + "=" * 80)
        print("Cleaning Transaction Fact Table (fact_txn)")
        print("=" * 80)
        
        df = self.df_transaction.copy()
        initial_count = len(df)
        stats = {}
        
        # 1. Date standardization
        print("\n1. Standardizing TXN_DT...")
        df['TXN_DT'] = pd.to_datetime(df['TXN_DT'], errors='coerce')
        
        # Check for invalid dates
        invalid_dates = df['TXN_DT'].isna()
        stats['invalid_dates'] = invalid_dates.sum()
        print(f"   - Found {stats['invalid_dates']} invalid dates")
        
        # Check for future dates
        future_dates = df['TXN_DT'] > datetime.now()
        stats['future_dates'] = future_dates.sum()
        print(f"   - Found {stats['future_dates']} future dates")
        
        if stats['future_dates'] > 0:
            # Set future dates to today
            df.loc[future_dates, 'TXN_DT'] = datetime.now()
            print(f"   - Adjusted future dates to current date")
        
        # 2. Handle missing PHYSICIAN_ID
        print("\n2. Handling missing PHYSICIAN_ID...")
        missing_physician = df['PHYSICIAN_ID'].isna()
        stats['missing_physician_id'] = missing_physician.sum()
        print(f"   - Found {stats['missing_physician_id']} missing PHYSICIAN_ID ({stats['missing_physician_id']/len(df)*100:.2f}%)")
        
        # Fill missing PHYSICIAN_ID with -1 (unknown marker)
        if stats['missing_physician_id'] > 0:
            df['PHYSICIAN_ID'] = df['PHYSICIAN_ID'].fillna(-1)
            print(f"   - Filled missing PHYSICIAN_ID with -1 (Unknown)")
        
        # Convert to int64 for consistency with dim_physician
        df['PHYSICIAN_ID'] = df['PHYSICIAN_ID'].astype('int64')
        print(f"   - Converted PHYSICIAN_ID to int64 for format consistency")
        
        # 3. Handle missing TXN_LOCATION_TYPE
        print("\n3. Handling missing TXN_LOCATION_TYPE...")
        
        # Handle blank values first
        df['TXN_LOCATION_TYPE'] = df['TXN_LOCATION_TYPE'].astype(str).str.strip()
        blank_location = df['TXN_LOCATION_TYPE'].isin(['', 'nan', 'None', 'NaN'])
        stats['missing_location'] = blank_location.sum()
        print(f"   - Found {stats['missing_location']} blank location types ({stats['missing_location']/len(df)*100:.2f}%)")
        
        if stats['missing_location'] > 0:
            df.loc[blank_location, 'TXN_LOCATION_TYPE'] = 'Unknown'
            print(f"   - Filled blank values with: Unknown")
        
        # 4. Text normalization
        print("\n4. Normalizing text fields...")
        
        # INSURANCE_TYPE
        if 'INSURANCE_TYPE' in df.columns:
            df['INSURANCE_TYPE'] = df['INSURANCE_TYPE'].astype(str).str.strip()
            df['INSURANCE_TYPE'] = df['INSURANCE_TYPE'].replace(['', 'nan', 'None', 'NaN'], 'Unknown')
            df['INSURANCE_TYPE'] = df['INSURANCE_TYPE'].str.upper()
            print(f"   - INSURANCE_TYPE: {df['INSURANCE_TYPE'].nunique()} unique values")
        
        # TXN_LOCATION_TYPE
        if 'TXN_LOCATION_TYPE' in df.columns:
            df['TXN_LOCATION_TYPE'] = df['TXN_LOCATION_TYPE'].str.upper()
            print(f"   - TXN_LOCATION_TYPE: {df['TXN_LOCATION_TYPE'].nunique()} unique values")
        
        # TXN_TYPE
        if 'TXN_TYPE' in df.columns:
            df['TXN_TYPE'] = df['TXN_TYPE'].astype(str).str.strip()
            df['TXN_TYPE'] = df['TXN_TYPE'].replace(['', 'nan', 'None', 'NaN'], 'Unknown')
            df['TXN_TYPE'] = df['TXN_TYPE'].str.title()
            print(f"   - TXN_TYPE: {df['TXN_TYPE'].nunique()} unique values")
        
        # TXN_DESC
        if 'TXN_DESC' in df.columns:
            df['TXN_DESC'] = df['TXN_DESC'].astype(str).str.strip()
            df['TXN_DESC'] = df['TXN_DESC'].replace(['', 'nan', 'None', 'NaN'], 'Unknown')
            df['TXN_DESC'] = df['TXN_DESC'].str.upper()
            print(f"   - TXN_DESC: {df['TXN_DESC'].nunique()} unique values")
        
        # 5. Validate IDs
        print("\n5. Validating ID fields...")
        
        # PATIENT_ID
        invalid_patient_id = (df['PATIENT_ID'] <= 0) | df['PATIENT_ID'].isna()
        stats['invalid_patient_id'] = invalid_patient_id.sum()
        print(f"   - Invalid PATIENT_ID: {stats['invalid_patient_id']}")
        
        # PHYSICIAN_ID (excluding -1 which marks unknown)
        invalid_physician_id = (df['PHYSICIAN_ID'] < -1) | (df['PHYSICIAN_ID'] == 0)
        stats['invalid_physician_id'] = invalid_physician_id.sum()
        print(f"   - Invalid PHYSICIAN_ID: {stats['invalid_physician_id']}")
        print(f"   - PHYSICIAN_ID = -1 (Unknown): {(df['PHYSICIAN_ID'] == -1).sum()}")
        
        # Remove records with invalid IDs
        if stats['invalid_patient_id'] > 0 or stats['invalid_physician_id'] > 0:
            df = df[~(invalid_patient_id | invalid_physician_id)]
            print(f"   - Removed {stats['invalid_patient_id'] + stats['invalid_physician_id']} records with invalid IDs")
        
        # 6. Check for duplicates
        print("\n6. Checking for duplicates...")
        duplicates = df.duplicated(keep='first')
        stats['duplicates'] = duplicates.sum()
        print(f"   - Found {stats['duplicates']} duplicate records")
        
        if stats['duplicates'] > 0:
            df = df[~duplicates]
            print(f"   - Removed duplicates")
        
        # 7. Sort by transaction date
        df = df.sort_values('TXN_DT').reset_index(drop=True)
        
        # Summary
        final_count = len(df)
        stats['records_removed'] = initial_count - final_count
        stats['final_count'] = final_count
        
        print("\n" + "-" * 80)
        print("Transaction Data Cleaning Summary:")
        print(f"  Initial records: {initial_count:,}")
        print(f"  Final records: {final_count:,}")
        print(f"  Records removed: {stats['records_removed']:,}")
        print(f"  Invalid dates fixed: {stats['invalid_dates']}")
        print(f"  Missing PHYSICIAN_ID: {stats['missing_physician_id']:,}")
        print(f"  Duplicates removed: {stats['duplicates']}")
        print("-" * 80)
        
        self.df_transaction = df
        self.cleaning_stats['transaction'] = stats
        
        return df
    
    def validate_referential_integrity(self):
        """Validate cross-table referential integrity"""
        print("\n" + "=" * 80)
        print("Validating Referential Integrity")
        print("=" * 80)
        
        stats = {}
        
        # 1. Check PATIENT_ID in transactions
        print("\n1. Checking PATIENT_ID referential integrity...")
        valid_patient_ids = set(self.df_patient['PATIENT_ID'].unique())
        transaction_patient_ids = set(self.df_transaction['PATIENT_ID'].unique())
        
        missing_patients = transaction_patient_ids - valid_patient_ids
        stats['missing_patients'] = len(missing_patients)
        print(f"   - Unique patients in transactions: {len(transaction_patient_ids):,}")
        print(f"   - Valid patients in dimension: {len(valid_patient_ids):,}")
        print(f"   - Missing patient IDs: {stats['missing_patients']}")
        
        if stats['missing_patients'] > 0:
            # Remove transactions with invalid patient IDs
            before_count = len(self.df_transaction)
            self.df_transaction = self.df_transaction[
                self.df_transaction['PATIENT_ID'].isin(valid_patient_ids)
            ]
            after_count = len(self.df_transaction)
            removed = before_count - after_count
            print(f"   - Removed {removed:,} transactions with invalid PATIENT_ID")
        
        # 2. Check PHYSICIAN_ID in transactions
        print("\n2. Checking PHYSICIAN_ID referential integrity...")
        valid_physician_ids = set(self.df_physician['PHYSICIAN_ID'].unique())
        
        # Check non-unknown physician IDs (-1 means unknown, which is valid)
        transaction_physician_ids = set(
            self.df_transaction[self.df_transaction['PHYSICIAN_ID'] != -1]['PHYSICIAN_ID'].unique()
        )
        
        missing_physicians = transaction_physician_ids - valid_physician_ids
        stats['missing_physicians'] = len(missing_physicians)
        unknown_physicians = (self.df_transaction['PHYSICIAN_ID'] == -1).sum()
        
        print(f"   - Unique physicians in transactions: {len(transaction_physician_ids):,}")
        print(f"   - Valid physicians in dimension: {len(valid_physician_ids):,}")
        print(f"   - Unknown physicians (-1): {unknown_physicians:,}")
        print(f"   - Missing physician IDs: {stats['missing_physicians']}")
        
        if stats['missing_physicians'] > 0:
            # Remove transactions with invalid physician IDs (keep -1 for unknown)
            before_count = len(self.df_transaction)
            mask = (
                self.df_transaction['PHYSICIAN_ID'].isin(valid_physician_ids) |
                (self.df_transaction['PHYSICIAN_ID'] == -1)
            )
            self.df_transaction = self.df_transaction[mask]
            after_count = len(self.df_transaction)
            removed = before_count - after_count
            print(f"   - Removed {removed:,} transactions with invalid PHYSICIAN_ID")
        
        # 3. Temporal logic check
        print("\n3. Checking temporal logic...")
        
        # Merge transaction with physician to get birth year
        txn_with_physician = self.df_transaction.merge(
            self.df_physician[['PHYSICIAN_ID', 'BIRTH_YEAR']],
            on='PHYSICIAN_ID',
            how='left'
        )
        
        # Calculate physician age at transaction
        txn_with_physician['PHYSICIAN_AGE'] = (
            txn_with_physician['TXN_DT'].dt.year - txn_with_physician['BIRTH_YEAR']
        )
        
        # Check if physician was at least 25 years old
        young_physician = (txn_with_physician['PHYSICIAN_AGE'] < 25) & txn_with_physician['PHYSICIAN_AGE'].notna()
        stats['young_physician_transactions'] = young_physician.sum()
        print(f"   - Transactions with physician < 25 years old: {stats['young_physician_transactions']}")
        
        if stats['young_physician_transactions'] > 0:
            print(f"   - Warning: Found transactions where physician may be too young")
            print(f"   - Keeping these records but flagging for review")
        
        print("\n" + "-" * 80)
        print("Referential Integrity Summary:")
        print(f"  Missing patient IDs: {stats['missing_patients']}")
        print(f"  Missing physician IDs: {stats['missing_physicians']}")
        print(f"  Young physician transactions: {stats['young_physician_transactions']}")
        print("-" * 80)
        
        return stats
    
    def save_cleaned_data(self):
        """Save cleaned data to CSV files"""
        print("\n" + "=" * 80)
        print("Saving Cleaned Data")
        print("=" * 80)
        
        try:
            # Save patient data
            patient_path = os.path.join(self.cleaned_dir, 'dim_patient_cleaned.csv')
            self.df_patient.to_csv(patient_path, index=False)
            print(f"✓ Saved: {patient_path}")
            print(f"  Shape: {self.df_patient.shape}")
            
            # Save physician data
            physician_path = os.path.join(self.cleaned_dir, 'dim_physician_cleaned.csv')
            self.df_physician.to_csv(physician_path, index=False)
            print(f"✓ Saved: {physician_path}")
            print(f"  Shape: {self.df_physician.shape}")
            
            # Save transaction data
            transaction_path = os.path.join(self.cleaned_dir, 'fact_txn_cleaned.csv')
            self.df_transaction.to_csv(transaction_path, index=False)
            print(f"✓ Saved: {transaction_path}")
            print(f"  Shape: {self.df_transaction.shape}")
            
            return True
        except Exception as e:
            print(f"✗ Error saving data: {str(e)}")
            return False
    
    def generate_cleaning_report(self):
        """Generate comprehensive cleaning report"""
        print("\n" + "=" * 80)
        print("DATA CLEANING REPORT")
        print("=" * 80)
        
        print("\n" + "─" * 80)
        print("1. PATIENT DIMENSION TABLE (dim_patient)")
        print("─" * 80)
        stats = self.cleaning_stats['patient']
        print(f"Initial Records:        {stats.get('final_count', 0) + stats.get('records_removed', 0):>10,}")
        print(f"Records Removed:        {stats.get('records_removed', 0):>10,}")
        print(f"Final Records:          {stats.get('final_count', 0):>10,}")
        print(f"Invalid Birth Years:    {stats.get('invalid_birth_year', 0):>10,}")
        print(f"Duplicates Removed:     {stats.get('duplicates', 0):>10,}")
        
        print("\n" + "─" * 80)
        print("2. PHYSICIAN DIMENSION TABLE (dim_physician)")
        print("─" * 80)
        stats = self.cleaning_stats['physician']
        print(f"Initial Records:        {stats.get('final_count', 0) + stats.get('records_removed', 0):>10,}")
        print(f"Records Removed:        {stats.get('records_removed', 0):>10,}")
        print(f"Final Records:          {stats.get('final_count', 0):>10,}")
        print(f"Invalid Birth Years:    {stats.get('invalid_birth_year', 0):>10,}")
        print(f"Duplicates Removed:     {stats.get('duplicates', 0):>10,}")
        
        print("\n" + "─" * 80)
        print("3. TRANSACTION FACT TABLE (fact_txn)")
        print("─" * 80)
        stats = self.cleaning_stats['transaction']
        print(f"Initial Records:        {stats.get('final_count', 0) + stats.get('records_removed', 0):>10,}")
        print(f"Records Removed:        {stats.get('records_removed', 0):>10,}")
        print(f"Final Records:          {stats.get('final_count', 0):>10,}")
        print(f"Invalid Dates:          {stats.get('invalid_dates', 0):>10,}")
        print(f"Missing Physician ID:   {stats.get('missing_physician_id', 0):>10,}")
        print(f"Duplicates Removed:     {stats.get('duplicates', 0):>10,}")
        
        print("\n" + "=" * 80)
        print("CLEANING COMPLETED SUCCESSFULLY")
        print("=" * 80)
    
    def assess_data_quality(self, df, table_name):
        """
        Assess data quality for a given dataframe
        
        Args:
            df: DataFrame to assess
            table_name: Name of the table for reporting
            
        Returns:
            dict: Quality assessment results
        """
        if df is None or len(df) == 0:
            return {}
        
        print(f"\nAssessing data quality for {table_name}...")
        
        quality_stats = {
            'table_name': table_name,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'total_cells': df.size,
            'missing_cells': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'completeness_score': 0,
            'consistency_score': 0,
            'validity_score': 0,
            'overall_score': 0,
            'column_details': {}
        }
        
        # Calculate completeness score
        if quality_stats['total_cells'] > 0:
            quality_stats['completeness_score'] = round(
                ((quality_stats['total_cells'] - quality_stats['missing_cells']) / 
                 quality_stats['total_cells'] * 100), 2
            )
        
        # Calculate consistency score (based on duplicate rate)
        duplicate_rate = (quality_stats['duplicate_records'] / quality_stats['total_records'] * 100) if quality_stats['total_records'] > 0 else 0
        quality_stats['consistency_score'] = round(max(0, 100 - duplicate_rate), 2)
        
        # Assess column-level quality
        for col in df.columns:
            col_stats = {
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': round((df[col].isnull().sum() / len(df) * 100), 2),
                'unique_values': df[col].nunique(),
                'unique_percentage': round((df[col].nunique() / len(df) * 100), 2)
            }
            
            # For numeric columns, add statistical measures
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': round(df[col].mean(), 2) if pd.notna(df[col].mean()) else None,
                    'median': df[col].median(),
                    'std': round(df[col].std(), 2) if pd.notna(df[col].std()) else None
                })
                
                # Check for outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                col_stats['outliers'] = outliers
                col_stats['outliers_percentage'] = round((outliers / len(df) * 100), 2)
            
            # For string columns, check format consistency
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                # Check for leading/trailing whitespace
                has_whitespace = df[col].notna() & (df[col].astype(str).str.strip() != df[col].astype(str))
                col_stats['whitespace_issues'] = has_whitespace.sum()
                
                # Check for mixed case (for columns with few unique values)
                if df[col].nunique() < 100:
                    lower_case = df[col].dropna().astype(str).str.lower()
                    col_stats['mixed_case'] = lower_case.nunique() < df[col].nunique()
            
            quality_stats['column_details'][col] = col_stats
        
        # Calculate validity score (based on outliers and format issues)
        validity_issues = 0
        for col_stats in quality_stats['column_details'].values():
            if col_stats.get('outliers_percentage', 0) > 10:
                validity_issues += 1
            if col_stats.get('whitespace_issues', 0) > 0:
                validity_issues += 0.5
            if col_stats.get('mixed_case', False):
                validity_issues += 0.5
        
        quality_stats['validity_score'] = round(max(0, 100 - (validity_issues * 10)), 2)
        
        # Calculate overall quality score (weighted average)
        quality_stats['overall_score'] = round(
            (quality_stats['completeness_score'] * 0.4 + 
             quality_stats['consistency_score'] * 0.3 + 
             quality_stats['validity_score'] * 0.3), 2
        )
        
        # Add quality grade
        score = quality_stats['overall_score']
        if score >= 90:
            quality_stats['quality_grade'] = 'A (Excellent)'
        elif score >= 80:
            quality_stats['quality_grade'] = 'B (Good)'
        elif score >= 70:
            quality_stats['quality_grade'] = 'C (Fair)'
        elif score >= 60:
            quality_stats['quality_grade'] = 'D (Poor)'
        else:
            quality_stats['quality_grade'] = 'F (Critical)'
        
        print(f"  Quality Score: {quality_stats['overall_score']}% ({quality_stats['quality_grade']})")
        
        return quality_stats
    
    def run_quality_assessment(self):
        """Run quality assessment on all cleaned datasets"""
        if not self.generate_quality_report:
            return
        
        print("\n" + "=" * 80)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 80)
        
        # Assess each table
        if self.df_patient is not None:
            self.quality_results['patient'] = self.assess_data_quality(self.df_patient, 'dim_patient')
        
        if self.df_physician is not None:
            self.quality_results['physician'] = self.assess_data_quality(self.df_physician, 'dim_physician')
        
        if self.df_transaction is not None:
            self.quality_results['transaction'] = self.assess_data_quality(self.df_transaction, 'fact_txn')
        
        print("\n" + "=" * 80)
        print("QUALITY ASSESSMENT COMPLETED")
        print("=" * 80)
    
    def _generate_quality_reports(self):
        """Generate comprehensive quality assessment reports"""
        if not self.generate_quality_report:
            return
        
        print("\n" + "=" * 80)
        print("GENERATING QUALITY ASSESSMENT REPORTS")
        print("=" * 80)
        
        # Create reports directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        reports_dir = os.path.join(base_dir, 'reports', 'data_quality')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate text report
        self._generate_text_quality_report(reports_dir)
        
        # Generate summary report
        self._generate_summary_quality_report(reports_dir)
        
        print(f"✓ Quality reports generated in: {reports_dir}")
    
    def _generate_text_quality_report(self, reports_dir):
        """Generate detailed text quality report"""
        report_path = os.path.join(reports_dir, 'data_quality_assessment_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA QUALITY ASSESSMENT REPORT\n")
            f.write("Pfizer EMR Alert System Dataset\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            overall_scores = []
            for table_name, results in self.quality_results.items():
                if results:
                    overall_scores.append(results['overall_score'])
                    f.write(f"{table_name.upper()} TABLE:\n")
                    f.write(f"  Quality Score: {results['overall_score']}% ({results['quality_grade']})\n")
                    f.write(f"  Total Records: {results['total_records']:,}\n")
                    f.write(f"  Completeness: {results['completeness_score']}%\n")
                    f.write(f"  Consistency: {results['consistency_score']}%\n")
                    f.write(f"  Validity: {results['validity_score']}%\n\n")
            
            if overall_scores:
                avg_score = sum(overall_scores) / len(overall_scores)
                f.write(f"OVERALL AVERAGE SCORE: {avg_score:.2f}%\n\n")
            
            # Detailed Analysis for each table
            for table_name, results in self.quality_results.items():
                if not results:
                    continue
                
                f.write(f"DETAILED ANALYSIS - {table_name.upper()} TABLE\n")
                f.write("=" * 50 + "\n")
                
                f.write(f"Table: {results['table_name']}\n")
                f.write(f"Total Records: {results['total_records']:,}\n")
                f.write(f"Total Columns: {results['total_columns']}\n")
                f.write(f"Total Cells: {results['total_cells']:,}\n")
                f.write(f"Missing Cells: {results['missing_cells']:,}\n")
                f.write(f"Duplicate Records: {results['duplicate_records']:,}\n")
                f.write(f"Overall Quality Score: {results['overall_score']}% ({results['quality_grade']})\n\n")
                
                f.write("COLUMN DETAILS:\n")
                f.write("-" * 30 + "\n")
                for col_name, col_stats in results['column_details'].items():
                    f.write(f"\nColumn: {col_name}\n")
                    f.write(f"  Data Type: {col_stats['dtype']}\n")
                    f.write(f"  Missing Values: {col_stats['missing_count']:,} ({col_stats['missing_percentage']}%)\n")
                    f.write(f"  Unique Values: {col_stats['unique_values']:,} ({col_stats['unique_percentage']}%)\n")
                    
                    if 'min' in col_stats:
                        f.write(f"  Min: {col_stats['min']}\n")
                        f.write(f"  Max: {col_stats['max']}\n")
                        f.write(f"  Mean: {col_stats['mean']}\n")
                        f.write(f"  Median: {col_stats['median']}\n")
                        f.write(f"  Std Dev: {col_stats['std']}\n")
                        f.write(f"  Outliers: {col_stats['outliers']:,} ({col_stats['outliers_percentage']}%)\n")
                    
                    if col_stats.get('whitespace_issues', 0) > 0:
                        f.write(f"  Whitespace Issues: {col_stats['whitespace_issues']:,}\n")
                    
                    if col_stats.get('mixed_case', False):
                        f.write(f"  Mixed Case Issues: Yes\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        print(f"✓ Detailed text report saved: {report_path}")
    
    def _generate_summary_quality_report(self, reports_dir):
        """Generate summary quality report"""
        summary_path = os.path.join(reports_dir, 'quality_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("DATA QUALITY SUMMARY\n")
            f.write("=" * 40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TABLE QUALITY SCORES:\n")
            f.write("-" * 30 + "\n")
            
            overall_scores = []
            for table_name, results in self.quality_results.items():
                if results:
                    overall_scores.append(results['overall_score'])
                    f.write(f"{table_name.upper():<15} {results['overall_score']:>6}% ({results['quality_grade']})\n")
            
            if overall_scores:
                avg_score = sum(overall_scores) / len(overall_scores)
                f.write(f"\nOVERALL AVERAGE: {avg_score:.2f}%\n")
                
                # Quality recommendations
                f.write("\nQUALITY RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                
                for table_name, results in self.quality_results.items():
                    if results and results['overall_score'] < 80:
                        f.write(f"\n{table_name.upper()} TABLE ISSUES:\n")
                        
                        if results['completeness_score'] < 90:
                            f.write(f"  - Low completeness ({results['completeness_score']}%): Review missing data\n")
                        
                        if results['consistency_score'] < 90:
                            f.write(f"  - Low consistency ({results['consistency_score']}%): Check for duplicates and format issues\n")
                        
                        if results['validity_score'] < 90:
                            f.write(f"  - Low validity ({results['validity_score']}%): Review outliers and data ranges\n")
        
        print(f"✓ Summary report saved: {summary_path}")
    
    def run(self):
        """Execute complete cleaning pipeline with integrated quality assessment"""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "DATA CLEANING PIPELINE" + " " * 36 + "║")
        print("║" + " " * 15 + "Pfizer EMR Alert System Dataset" + " " * 31 + "║")
        print("╚" + "═" * 78 + "╝")
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Clean individual tables
        self.clean_patient_data()
        self.clean_physician_data()
        self.clean_transaction_data()
        
        # Step 3: Validate referential integrity
        self.validate_referential_integrity()
        
        # Step 4: Save cleaned data
        if not self.save_cleaned_data():
            return False
        
        # Step 5: Generate cleaning report
        self.generate_cleaning_report()
        
        # Step 6: Run quality assessment (if enabled)
        self.run_quality_assessment()
        
        # Step 7: Generate quality reports (if enabled)
        self._generate_quality_reports()
        
        return True


def main():
    """Main function with integrated quality assessment"""
    # Setup paths - 
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    cleaned_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Create cleaner instance with quality assessment enabled by default
    cleaner = DataCleaner(raw_dir, cleaned_dir, generate_quality_report=True)
    
    # Run cleaning pipeline with integrated quality assessment
    success = cleaner.run()
    
    if success:
        print("\n✓ Data cleaning completed successfully!")
        print(f"✓ Cleaned files saved to: {cleaned_dir}")
        print("✓ Quality assessment reports generated in: reports/data_quality/")
        print("\nGenerated reports:")
        print("  - data_quality_assessment_report.txt (detailed analysis)")
        print("  - quality_summary.txt (executive summary)")
    else:
        print("\n✗ Data cleaning failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

