"""
Data Quality Assessment Report Generator

This script reads raw datasets and generates comprehensive data quality assessment reports in PDF format.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class DataQualityReport:
    """Data Quality Report Generator Class"""
    
    def __init__(self, data_path, output_dir, dataset_name):
        """
        Initialize Data Quality Report
        
        Args:
            data_path: Path to data file
            output_dir: Output directory
            dataset_name: Name of the dataset
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.df = None
        self.quality_results = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def read_data(self):
        """Read data file"""
        try:
            file_extension = os.path.splitext(self.data_path)[1].lower()
            
            if file_extension == '.csv':
                self.df = pd.read_csv(self.data_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            print(f"Successfully loaded data: {self.data_path}")
            print(f"Data shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Failed to load data: {str(e)}")
            return False
    
    def assess_completeness(self):
        """Assess data completeness"""
        completeness = {}
        
        # Total records
        total_rows = len(self.df)
        total_cells = self.df.size
        
        # Missing value statistics
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / total_rows * 100).round(2)
        
        # Column-wise statistics
        column_stats = []
        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            null_count = self.df[col].isna().sum()
            null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
            
            column_stats.append({
                'column': col,
                'total_records': total_rows,
                'non_null_count': non_null,
                'null_count': null_count,
                'null_percentage': round(null_pct, 2),
                'completeness': round(100 - null_pct, 2)
            })
        
        completeness['column_stats'] = column_stats
        completeness['total_rows'] = total_rows
        completeness['total_columns'] = len(self.df.columns)
        completeness['total_cells'] = total_cells
        completeness['total_missing'] = missing_counts.sum()
        completeness['overall_completeness'] = round(
            ((total_cells - missing_counts.sum()) / total_cells * 100) if total_cells > 0 else 0, 2
        )
        
        return completeness
    
    def assess_validity(self):
        """Assess data validity"""
        validity = {}
        
        for col in self.df.columns:
            col_validity = {
                'column': col,
                'dtype': str(self.df[col].dtype),
                'unique_values': self.df[col].nunique(),
                'unique_percentage': round((self.df[col].nunique() / len(self.df) * 100), 2)
            }
            
            # For numeric types, calculate statistical information
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_validity['min'] = self.df[col].min()
                col_validity['max'] = self.df[col].max()
                col_validity['mean'] = round(self.df[col].mean(), 2) if pd.notna(self.df[col].mean()) else None
                col_validity['median'] = self.df[col].median()
                col_validity['std'] = round(self.df[col].std(), 2) if pd.notna(self.df[col].std()) else None
                
                # Check outliers (using IQR method)
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                col_validity['outliers'] = outliers
                col_validity['outliers_percentage'] = round((outliers / len(self.df) * 100), 2)
            
            # For string types
            elif pd.api.types.is_string_dtype(self.df[col]) or self.df[col].dtype == 'object':
                # Calculate most common value
                if self.df[col].notna().any():
                    value_counts = self.df[col].value_counts()
                    col_validity['most_common_value'] = value_counts.index[0] if len(value_counts) > 0 else None
                    col_validity['most_common_count'] = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            
            # For datetime types
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_validity['min_date'] = str(self.df[col].min())
                col_validity['max_date'] = str(self.df[col].max())
            
            validity[col] = col_validity
        
        return validity
    
    def assess_consistency(self):
        """Assess data consistency"""
        consistency = {
            'duplicates': {
                'total_duplicates': self.df.duplicated().sum(),
                'duplicate_percentage': round((self.df.duplicated().sum() / len(self.df) * 100), 2)
            }
        }
        
        # Check format consistency for each column (for string types)
        format_issues = []
        for col in self.df.columns:
            if pd.api.types.is_string_dtype(self.df[col]) or self.df[col].dtype == 'object':
                # Check for leading/trailing whitespace
                has_whitespace = self.df[col].notna() & (self.df[col].astype(str).str.strip() != self.df[col].astype(str))
                if has_whitespace.any():
                    format_issues.append({
                        'column': col,
                        'issue': 'Leading/trailing whitespace',
                        'affected_rows': has_whitespace.sum()
                    })
                
                # Check for inconsistent case (for columns with few unique values)
                if self.df[col].nunique() < 100:
                    lower_case = self.df[col].dropna().astype(str).str.lower()
                    if lower_case.nunique() < self.df[col].nunique():
                        format_issues.append({
                            'column': col,
                            'issue': 'Mixed case',
                            'affected_rows': len(self.df[col].dropna())
                        })
        
        consistency['format_issues'] = format_issues
        
        return consistency
    
    def assess_accuracy(self):
        """Assess data accuracy (based on business rules)"""
        accuracy = {}
        
        # Business rule checks for specific fields
        rules_violations = []
        
        # Example: Check if year is within reasonable range
        for col in self.df.columns:
            if 'year' in col.lower() or 'yr' in col.lower():
                current_year = datetime.now().year
                invalid_years = ((self.df[col] < 1900) | (self.df[col] > current_year)).sum()
                if invalid_years > 0:
                    rules_violations.append({
                        'column': col,
                        'rule': 'Year should be between 1900 and current year',
                        'violations': invalid_years
                    })
            
            # Check age-related fields
            if 'age' in col.lower():
                invalid_age = ((self.df[col] < 0) | (self.df[col] > 120)).sum()
                if invalid_age > 0:
                    rules_violations.append({
                        'column': col,
                        'rule': 'Age should be between 0 and 120',
                        'violations': invalid_age
                    })
        
        accuracy['rules_violations'] = rules_violations
        
        return accuracy
    
    def assess_timeliness(self):
        """Assess data timeliness"""
        timeliness = {}
        
        # Find date columns
        date_columns = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                date_columns.append(col)
            elif 'date' in col.lower() or 'dt' in col.lower():
                try:
                    temp_dates = pd.to_datetime(self.df[col], errors='coerce')
                    if temp_dates.notna().sum() > 0:
                        date_columns.append(col)
                except:
                    pass
        
        if date_columns:
            for col in date_columns:
                try:
                    dates = pd.to_datetime(self.df[col], errors='coerce')
                    if dates.notna().any():
                        timeliness[col] = {
                            'earliest_date': str(dates.min()),
                            'latest_date': str(dates.max()),
                            'date_range_days': (dates.max() - dates.min()).days if pd.notna(dates.max()) and pd.notna(dates.min()) else 0
                        }
                except:
                    pass
        
        return timeliness
    
    def generate_summary(self):
        """Generate data quality summary"""
        completeness = self.quality_results['completeness']
        validity = self.quality_results['validity']
        consistency = self.quality_results['consistency']
        
        # Calculate quality scores
        completeness_score = completeness['overall_completeness']
        
        # Consistency score (based on duplicate rate)
        consistency_score = max(0, 100 - consistency['duplicates']['duplicate_percentage'])
        
        # Validity score (based on significant outliers)
        validity_issues = sum(1 for col_info in validity.values() 
                             if col_info.get('outliers_percentage', 0) > 10)
        validity_score = max(0, 100 - (validity_issues * 10))
        
        # Overall quality score (weighted average)
        overall_score = (completeness_score * 0.4 + 
                        consistency_score * 0.3 + 
                        validity_score * 0.3)
        
        summary = {
            'completeness_score': round(completeness_score, 2),
            'consistency_score': round(consistency_score, 2),
            'validity_score': round(validity_score, 2),
            'overall_score': round(overall_score, 2),
            'quality_grade': self._get_grade(overall_score)
        }
        
        return summary
    
    def _get_grade(self, score):
        """Return grade based on score"""
        if score >= 90:
            return 'A (Excellent)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Fair)'
        elif score >= 60:
            return 'D (Poor)'
        else:
            return 'F (Critical)'
    
    def run_assessment(self):
        """Execute complete data quality assessment"""
        if not self.read_data():
            return False
        
        print("Starting data quality assessment...")
        
        # Execute assessments
        self.quality_results['completeness'] = self.assess_completeness()
        print("- Completeness assessment completed")
        
        self.quality_results['validity'] = self.assess_validity()
        print("- Validity assessment completed")
        
        self.quality_results['consistency'] = self.assess_consistency()
        print("- Consistency assessment completed")
        
        self.quality_results['accuracy'] = self.assess_accuracy()
        print("- Accuracy assessment completed")
        
        self.quality_results['timeliness'] = self.assess_timeliness()
        print("- Timeliness assessment completed")
        
        self.quality_results['summary'] = self.generate_summary()
        print("- Quality summary generated")
        
        return True
    
    def create_visualizations(self):
        """Create visualization charts"""
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Completeness bar chart
        completeness = self.quality_results['completeness']
        column_stats = completeness['column_stats']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        columns = [stat['column'] for stat in column_stats]
        completeness_pct = [stat['completeness'] for stat in column_stats]
        
        bars = ax.bar(columns, completeness_pct, color='steelblue')
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Target (90%)')
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_ylabel('Completeness (%)', fontsize=12)
        ax.set_title(f'Data Completeness by Column - {self.dataset_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        
        # Display percentages on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        completeness_chart = os.path.join(viz_dir, f'{self.dataset_name}_completeness.png')
        plt.savefig(completeness_chart, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Quality dimensions bar chart
        summary = self.quality_results['summary']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Completeness', 'Consistency', 'Validity']
        scores = [
            summary['completeness_score'],
            summary['consistency_score'],
            summary['validity_score']
        ]
        
        x = np.arange(len(categories))
        bars = ax.bar(x, scores, color=['#4CAF50', '#2196F3', '#FF9800'])
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title(f'Data Quality Dimensions - {self.dataset_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 105)
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Threshold (80%)')
        ax.legend()
        
        # Display scores on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        scores_chart = os.path.join(viz_dir, f'{self.dataset_name}_scores.png')
        plt.savefig(scores_chart, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'completeness_chart': completeness_chart,
            'scores_chart': scores_chart
        }
    
    def generate_pdf_report(self):
        """Generate PDF report"""
        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_data_quality_report.pdf')
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for all elements
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1976d2'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        # Title
        elements.append(Paragraph(f"Data Quality Assessment Report", title_style))
        elements.append(Paragraph(f"Dataset: {self.dataset_name}", styles['Heading2']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Executive summary
        elements.append(Paragraph("Executive Summary", heading_style))
        summary = self.quality_results['summary']
        completeness = self.quality_results['completeness']
        
        summary_data = [
            ['Metric', 'Value'],
            ['Overall Quality Score', f"{summary['overall_score']}%"],
            ['Quality Grade', summary['quality_grade']],
            ['Total Records', f"{completeness['total_rows']:,}"],
            ['Total Columns', str(completeness['total_columns'])],
            ['Completeness Score', f"{summary['completeness_score']}%"],
            ['Consistency Score', f"{summary['consistency_score']}%"],
            ['Validity Score', f"{summary['validity_score']}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Visualization charts
        charts = self.create_visualizations()
        
        elements.append(Paragraph("Data Quality Visualization", heading_style))
        if os.path.exists(charts['scores_chart']):
            img = Image(charts['scores_chart'], width=5*inch, height=3.75*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        
        elements.append(PageBreak())
        
        # Completeness details
        elements.append(Paragraph("1. Data Completeness Analysis", heading_style))
        
        column_stats = completeness['column_stats']
        completeness_data = [['Column', 'Total Records', 'Non-Null', 'Null Count', 'Null %', 'Completeness']]
        
        for stat in column_stats:
            completeness_data.append([
                stat['column'],
                f"{stat['total_records']:,}",
                f"{stat['non_null_count']:,}",
                f"{stat['null_count']:,}",
                f"{stat['null_percentage']}%",
                f"{stat['completeness']}%"
            ])
        
        completeness_table = Table(completeness_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 0.8*inch, 1*inch])
        completeness_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        elements.append(completeness_table)
        elements.append(Spacer(1, 0.2*inch))
        
        if os.path.exists(charts['completeness_chart']):
            img = Image(charts['completeness_chart'], width=6.5*inch, height=3.25*inch)
            elements.append(img)
        
        elements.append(PageBreak())
        
        # Consistency details
        elements.append(Paragraph("2. Data Consistency Analysis", heading_style))
        consistency = self.quality_results['consistency']
        
        consistency_text = f"""
        <b>Duplicate Records:</b> {consistency['duplicates']['total_duplicates']:,} 
        ({consistency['duplicates']['duplicate_percentage']}%)<br/>
        """
        elements.append(Paragraph(consistency_text, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        if consistency['format_issues']:
            elements.append(Paragraph("<b>Format Issues Detected:</b>", styles['Normal']))
            format_data = [['Column', 'Issue', 'Affected Rows']]
            for issue in consistency['format_issues']:
                format_data.append([
                    issue['column'],
                    issue['issue'],
                    str(issue['affected_rows'])
                ])
            
            format_table = Table(format_data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
            format_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            elements.append(format_table)
        else:
            elements.append(Paragraph("No format issues detected.", styles['Normal']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Validity details
        elements.append(Paragraph("3. Data Validity Analysis", heading_style))
        validity = self.quality_results['validity']
        
        validity_data = [['Column', 'Data Type', 'Unique Values', 'Unique %']]
        for col, info in validity.items():
            validity_data.append([
                info['column'],
                info['dtype'],
                f"{info['unique_values']:,}",
                f"{info['unique_percentage']}%"
            ])
        
        validity_table = Table(validity_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        validity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        elements.append(validity_table)
        elements.append(PageBreak())
        
        # Statistical summary for numeric columns
        elements.append(Paragraph("Statistical Summary for Numeric Columns", heading_style))
        numeric_data = [['Column', 'Min', 'Max', 'Mean', 'Median', 'Std Dev', 'Outliers']]
        
        for col, info in validity.items():
            if 'min' in info:
                numeric_data.append([
                    info['column'],
                    f"{info.get('min', 'N/A')}",
                    f"{info.get('max', 'N/A')}",
                    f"{info.get('mean', 'N/A')}",
                    f"{info.get('median', 'N/A')}",
                    f"{info.get('std', 'N/A')}",
                    f"{info.get('outliers', 0)} ({info.get('outliers_percentage', 0)}%)"
                ])
        
        if len(numeric_data) > 1:
            numeric_table = Table(numeric_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.2*inch])
            numeric_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            elements.append(numeric_table)
        else:
            elements.append(Paragraph("No numeric columns found in the dataset.", styles['Normal']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Accuracy analysis
        accuracy = self.quality_results['accuracy']
        if accuracy['rules_violations']:
            elements.append(Paragraph("4. Data Accuracy - Business Rules Violations", heading_style))
            
            rules_data = [['Column', 'Rule', 'Violations']]
            for violation in accuracy['rules_violations']:
                rules_data.append([
                    violation['column'],
                    violation['rule'],
                    str(violation['violations'])
                ])
            
            rules_table = Table(rules_data, colWidths=[1.5*inch, 3*inch, 1*inch])
            rules_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            elements.append(rules_table)
        
        # Timeliness analysis
        timeliness = self.quality_results['timeliness']
        if timeliness:
            elements.append(Spacer(1, 0.3*inch))
            elements.append(Paragraph("5. Data Timeliness Analysis", heading_style))
            
            time_data = [['Column', 'Earliest Date', 'Latest Date', 'Date Range (Days)']]
            for col, info in timeliness.items():
                time_data.append([
                    col,
                    info['earliest_date'],
                    info['latest_date'],
                    f"{info['date_range_days']:,}"
                ])
            
            time_table = Table(time_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            time_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            elements.append(time_table)
        
        # Build PDF
        doc.build(elements)
        print(f"\n✓ PDF report generated: {output_path}")
        
        return output_path


def main():
    """Main function"""
    # Setup paths - 修正路径配置以匹配项目结构
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    output_dir = os.path.join(base_dir, 'reports', 'data_quality')
    
    # List of data files
    datasets = [
        {
            'file': os.path.join(raw_data_dir, 'dim_patient.xlsx'),
            'name': 'dim_patient'
        },
        {
            'file': os.path.join(raw_data_dir, 'dim_physician.xlsx'),
            'name': 'dim_physician'
        },
        {
            'file': os.path.join(raw_data_dir, 'fact_txn.xlsx'),
            'name': 'fact_txn'
        }
    ]
    
    print("=" * 80)
    print("Data Quality Assessment Report Generator")
    print("=" * 80)
    print()
    
    # Generate reports for each dataset
    for dataset in datasets:
        if not os.path.exists(dataset['file']):
            print(f"⚠ Warning: File not found - {dataset['file']}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset['name']}")
        print(f"{'=' * 80}")
        
        # Create report generator
        reporter = DataQualityReport(
            data_path=dataset['file'],
            output_dir=output_dir,
            dataset_name=dataset['name']
        )
        
        # Execute assessment
        if reporter.run_assessment():
            # Generate PDF report
            reporter.generate_pdf_report()
            print(f"✓ {dataset['name']} data quality report completed!")
        else:
            print(f"✗ {dataset['name']} data quality assessment failed!")
    
    print("\n" + "=" * 80)
    print("All reports generated successfully!")
    print(f"Report location: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

