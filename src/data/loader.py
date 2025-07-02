"""
Data loader utilities for InfraOpt simulation.

This module provides utilities for loading external datasets
and integrating with real-world data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import os
import json
import logging

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Data loader for external datasets and real-world data sources.
    
    This class provides utilities for loading and preprocessing
    various types of data used in the simulation.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.logger = logger
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
        
        self.logger.info(f"DataLoader initialized with data directory: {data_dir}")
    
    def load_energy_data(self, file_path: str) -> pd.DataFrame:
        """
        Load energy market data.
        
        Args:
            file_path: Path to energy data file
            
        Returns:
            DataFrame containing energy data
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Loaded energy data from {file_path}: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load energy data from {file_path}: {str(e)}")
            raise
    
    def load_supply_chain_data(self, file_path: str) -> pd.DataFrame:
        """
        Load supply chain data.
        
        Args:
            file_path: Path to supply chain data file
            
        Returns:
            DataFrame containing supply chain data
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Loaded supply chain data from {file_path}: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load supply chain data from {file_path}: {str(e)}")
            raise
    
    def load_workload_data(self, file_path: str) -> pd.DataFrame:
        """
        Load workload pattern data.
        
        Args:
            file_path: Path to workload data file
            
        Returns:
            DataFrame containing workload data
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Loaded workload data from {file_path}: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load workload data from {file_path}: {str(e)}")
            raise
    
    def load_infrastructure_data(self, file_path: str) -> pd.DataFrame:
        """
        Load infrastructure specifications data.
        
        Args:
            file_path: Path to infrastructure data file
            
        Returns:
            DataFrame containing infrastructure data
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Loaded infrastructure data from {file_path}: {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load infrastructure data from {file_path}: {str(e)}")
            raise
    
    def save_processed_data(self, data: pd.DataFrame, filename: str) -> str:
        """
        Save processed data to file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.data_dir, "processed", filename)
        
        try:
            if filename.endswith('.csv'):
                data.to_csv(output_path, index=False)
            elif filename.endswith('.json'):
                data.to_json(output_path, orient='records', indent=2)
            elif filename.endswith('.parquet'):
                data.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {filename}")
            
            self.logger.info(f"Saved processed data to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data to {output_path}: {str(e)}")
            raise
    
    def load_configuration(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame, schema: Dict[str, Any]) -> bool:
        """
        Validate data against schema.
        
        Args:
            data: DataFrame to validate
            schema: Schema definition
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check required columns
            required_columns = schema.get('required_columns', [])
            missing_columns = set(required_columns) - set(data.columns)
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check data types
            for column, expected_type in schema.get('column_types', {}).items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if actual_type != expected_type:
                        self.logger.warning(f"Column {column} has type {actual_type}, expected {expected_type}")
            
            # Check value ranges
            for column, range_info in schema.get('value_ranges', {}).items():
                if column in data.columns:
                    min_val = range_info.get('min')
                    max_val = range_info.get('max')
                    
                    if min_val is not None and data[column].min() < min_val:
                        self.logger.warning(f"Column {column} has values below minimum {min_val}")
                    
                    if max_val is not None and data[column].max() > max_val:
                        self.logger.warning(f"Column {column} has values above maximum {max_val}")
            
            self.logger.info("Data validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def preprocess_energy_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess energy market data.
        
        Args:
            data: Raw energy data
            
        Returns:
            Preprocessed energy data
        """
        try:
            # Handle missing values
            data = data.fillna(method='ffill')
            
            # Convert timestamps
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Calculate derived metrics
            if 'price' in data.columns and 'volume' in data.columns:
                data['total_value'] = data['price'] * data['volume']
            
            # Add time-based features
            if 'timestamp' in data.columns:
                data['hour'] = data['timestamp'].dt.hour
                data['day_of_week'] = data['timestamp'].dt.dayofweek
                data['month'] = data['timestamp'].dt.month
            
            self.logger.info("Energy data preprocessing completed")
            return data
            
        except Exception as e:
            self.logger.error(f"Energy data preprocessing failed: {str(e)}")
            raise
    
    def preprocess_workload_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess workload data.
        
        Args:
            data: Raw workload data
            
        Returns:
            Preprocessed workload data
        """
        try:
            # Handle missing values
            data = data.fillna(0)
            
            # Normalize resource requirements
            resource_columns = ['compute_intensity', 'memory_requirement', 'storage_requirement']
            for col in resource_columns:
                if col in data.columns:
                    data[col] = data[col].clip(lower=0)
            
            # Calculate workload complexity score
            if all(col in data.columns for col in resource_columns):
                data['complexity_score'] = (
                    data['compute_intensity'] * 0.4 +
                    data['memory_requirement'] * 0.3 +
                    data['storage_requirement'] * 0.3
                )
            
            self.logger.info("Workload data preprocessing completed")
            return data
            
        except Exception as e:
            self.logger.error(f"Workload data preprocessing failed: {str(e)}")
            raise
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame], merge_key: str) -> pd.DataFrame:
        """
        Merge multiple datasets on a common key.
        
        Args:
            datasets: Dictionary of datasets to merge
            merge_key: Key column for merging
            
        Returns:
            Merged dataset
        """
        try:
            if len(datasets) < 2:
                return list(datasets.values())[0] if datasets else pd.DataFrame()
            
            # Start with the first dataset
            merged = list(datasets.values())[0]
            
            # Merge with remaining datasets
            for name, dataset in list(datasets.items())[1:]:
                if merge_key in merged.columns and merge_key in dataset.columns:
                    merged = merged.merge(dataset, on=merge_key, how='left')
                else:
                    self.logger.warning(f"Merge key '{merge_key}' not found in dataset '{name}'")
            
            self.logger.info(f"Merged {len(datasets)} datasets")
            return merged
            
        except Exception as e:
            self.logger.error(f"Dataset merging failed: {str(e)}")
            raise
    
    def export_to_excel(self, data: pd.DataFrame, filename: str, sheets: Dict[str, pd.DataFrame] = None) -> str:
        """
        Export data to Excel file.
        
        Args:
            data: Main DataFrame to export
            filename: Output filename
            sheets: Additional sheets to include
            
        Returns:
            Path to exported file
        """
        output_path = os.path.join(self.data_dir, "processed", filename)
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name='Main', index=False)
                
                if sheets:
                    for sheet_name, sheet_data in sheets.items():
                        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            self.logger.info(f"Exported data to Excel: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export to Excel: {str(e)}")
            raise 