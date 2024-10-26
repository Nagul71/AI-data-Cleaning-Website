from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from scipy import stats

class DataCleaningRecommender:
    def __init__(self):
        self.cleaning_operations = {
            'missing_data': {
                'name': 'Handle Missing Data',
                'description': 'Fill or remove missing values',
                'priority': 0
            },
            'outliers': {
                'name': 'Handle Outliers',
                'description': 'Detect and handle statistical outliers',
                'priority': 0
            },
            'duplicates': {
                'name': 'Remove Duplicates',
                'description': 'Remove duplicate rows',
                'priority': 0
            },
            'encoding': {
                'name': 'Encode Categorical Data',
                'description': 'Encode categorical variables',
                'priority': 0
            },
            'normalization': {
                'name': 'Normalize Data',
                'description': 'Normalize numeric columns',
                'priority': 0
            },
            'date_conversion': {
                'name': 'Convert Dates',
                'description': 'Detect and convert date columns',
                'priority': 0
            }
        }

    def analyze_dataset(self, df):
        recommendations = []
        total_score = 0
        
        # Get dataset characteristics
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        duplicate_ratio = df.duplicated().sum() / len(df)
        categorical_cols = df.select_dtypes(include=['object']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Check for missing values
        if missing_ratio > 0:
            priority = self._calculate_priority(missing_ratio, 0.05, 0.3)
            self.cleaning_operations['missing_data']['priority'] = priority
            total_score += priority
            recommendations.append(self.cleaning_operations['missing_data'])

        # Check for duplicates
        if duplicate_ratio > 0:
            priority = self._calculate_priority(duplicate_ratio, 0.01, 0.1)
            self.cleaning_operations['duplicates']['priority'] = priority
            total_score += priority
            recommendations.append(self.cleaning_operations['duplicates'])

        # Check for outliers in numeric columns
        if len(numeric_cols) > 0:
            outlier_ratio = self._check_outliers(df[numeric_cols])
            if outlier_ratio > 0:
                priority = self._calculate_priority(outlier_ratio, 0.05, 0.2)
                self.cleaning_operations['outliers']['priority'] = priority
                total_score += priority
                recommendations.append(self.cleaning_operations['outliers'])

        # Check for categorical columns that might need encoding
        if len(categorical_cols) > 0:
            if self._need_encoding(df[categorical_cols]):
                self.cleaning_operations['encoding']['priority'] = 0.7
                total_score += 0.7
                recommendations.append(self.cleaning_operations['encoding'])

        # Check for numeric columns that might need normalization
        if len(numeric_cols) > 0:
            if self._need_normalization(df[numeric_cols]):
                self.cleaning_operations['normalization']['priority'] = 0.6
                total_score += 0.6
                recommendations.append(self.cleaning_operations['normalization'])

        # Check for potential date columns
        if self._detect_date_columns(df):
            self.cleaning_operations['date_conversion']['priority'] = 0.8
            total_score += 0.8
            recommendations.append(self.cleaning_operations['date_conversion'])

        # Sort recommendations by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations, total_score

    def _calculate_priority(self, ratio, low_threshold, high_threshold):
        if ratio < low_threshold:
            return 0.3
        elif ratio < high_threshold:
            return 0.7
        else:
            return 1.0

    def _check_outliers(self, numeric_data):
        outlier_counts = 0
        total_values = 0
        
        for column in numeric_data.columns:
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (numeric_data[column] < (Q1 - 1.5 * IQR)) | (numeric_data[column] > (Q3 + 1.5 * IQR))
            outlier_counts += outlier_mask.sum()
            total_values += len(numeric_data[column])
        
        return outlier_counts / total_values if total_values > 0 else 0

    def _need_encoding(self, categorical_data):
        unique_counts = categorical_data.nunique()
        return any(unique_counts > 2) and any(unique_counts < len(categorical_data) * 0.5)

    def _need_normalization(self, numeric_data):
        for column in numeric_data.columns:
            if abs(stats.skew(numeric_data[column].dropna())) > 1 or \
               abs(stats.kurtosis(numeric_data[column].dropna())) > 1:
                return True
        return False

    def _detect_date_columns(self, df):
        for column in df.select_dtypes(include=['object']):
            try:
                pd.to_datetime(df[column].iloc[0])
                return True
            except:
                continue
        return False

    def get_recommendation_details(self, operation):
        if operation in self.cleaning_operations:
            return self.cleaning_operations[operation]
        return None