"""
Data processing module for the EMR Alert System

This module contains data preprocessing and feature engineering utilities
that can be used by both the API service and training scripts.
"""

from .data_processor import DataProcessor

__all__ = ['DataProcessor']
