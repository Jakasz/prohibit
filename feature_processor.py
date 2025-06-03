# feature_processor.py
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import List, Tuple, Optional
import logging


class AdvancedFeatureProcessor:
    """Покращена обробка ознак"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Імпутери
        self.numeric_imputer = None
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        # Енкодери
        self.label_encoders = {}
        self.onehot_encoder = None
        
        # Списки ознак
        self.numeric_features = []
        self.categorical_features = []
        self.target_encoded_features = []
        
        # Статистики для target encoding
        self.target_encoding_stats = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Навчання процесора"""
        # Визначення типів ознак
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Налаштування імпутера для числових даних
        if len(self.numeric_features) > 5:
            self.numeric_imputer = KNNImputer(n_neighbors=5)
        else:
            self.numeric_imputer = SimpleImputer(strategy='median')
        
        # Target encoding для високо-кардинальних категоріальних змінних
        if y is not None:
            for col in self.categorical_features:
                if X[col].nunique() > 10:  # Високо-кардинальна змінна
                    self.target_encoded_features.append(col)
                    self._fit_target_encoding(X[col], y, col)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Трансформація даних"""
        X = X.copy()
        
        # Обробка числових ознак
        if self.numeric_features and self.numeric_imputer:
            X[self.numeric_features] = self.numeric_imputer.transform(X[self.numeric_features])
        
        # Обробка категоріальних ознак
        for col in self.categorical_features:
            if col in self.target_encoded_features:
                X[col] = self._apply_target_encoding(X[col], col)
            else:
                # Label encoding для низько-кардинальних
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].fillna('missing'))
                else:
                    # Обробка невідомих категорій
                    X[col] = X[col].fillna('missing')
                    X[col] = X[col].apply(lambda x: self._safe_label_encode(x, col))
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit та transform за один крок"""
        return self.fit(X, y).transform(X)
    
    def _fit_target_encoding(self, series: pd.Series, target: pd.Series, col_name: str):
        """Навчання target encoding"""
        # Глобальне середнє
        global_mean = target.mean()
        
        # Статистики по категоріях
        stats = pd.DataFrame({
            'sum': target.groupby(series).sum(),
            'count': target.groupby(series).count()
        })
        
        # Згладжування (smoothing)
        smoothing = 10
        stats['smoothed_mean'] = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        self.target_encoding_stats[col_name] = {
            'stats': stats,
            'global_mean': global_mean
        }
    
    def _apply_target_encoding(self, series: pd.Series, col_name: str) -> pd.Series:
        """Застосування target encoding"""
        stats = self.target_encoding_stats[col_name]['stats']
        global_mean = self.target_encoding_stats[col_name]['global_mean']
        
        # Мапінг значень
        encoded = series.map(stats['smoothed_mean'])
        
        # Заповнення пропусків глобальним середнім
        encoded = encoded.fillna(global_mean)
        
        return encoded
    
    def _safe_label_encode(self, value, col_name: str):
        """Безпечне label encoding з обробкою невідомих значень"""
        try:
            return self.label_encoders[col_name].transform([value])[0]
        except ValueError:
            # Невідома категорія - повертаємо -1
            return -1