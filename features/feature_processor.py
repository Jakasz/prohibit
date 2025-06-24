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
        
        # Прапорець fitted
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Навчання процесора"""
        # Визначення типів ознак
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Налаштування імпутера для числових даних
        if self.numeric_features:
            if len(self.numeric_features) > 5 and len(X) > 10:  # Достатньо даних для KNN
                self.numeric_imputer = KNNImputer(n_neighbors=min(5, len(X)-1))
            else:
                self.numeric_imputer = SimpleImputer(strategy='median')
            
            # Навчання числового імпутера
            if self.numeric_features:
                self.numeric_imputer.fit(X[self.numeric_features])
        
        # Навчання категоріального імпутера
        if self.categorical_features:
            self.categorical_imputer.fit(X[self.categorical_features])
        
        # Target encoding для високо-кардинальних категоріальних змінних
        if y is not None and self.categorical_features:
            for col in self.categorical_features:
                if col in X.columns and X[col].nunique() > 10:  # Високо-кардинальна змінна
                    self.target_encoded_features.append(col)
                    self._fit_target_encoding(X[col], y, col)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Трансформація даних"""
        if not self.is_fitted:
            raise RuntimeError("FeatureProcessor not fitted. Call fit() first.")
        
        X = X.copy()
        
        # Обробка числових ознак
        if self.numeric_features and self.numeric_imputer:
            # Перевірка наявності числових колонок
            existing_numeric = [col for col in self.numeric_features if col in X.columns]
            if existing_numeric:
                X[existing_numeric] = self.numeric_imputer.transform(X[existing_numeric])
        
        # Обробка категоріальних ознак
        if self.categorical_features:
            # Перевірка наявності категоріальних колонок
            existing_categorical = [col for col in self.categorical_features if col in X.columns]
            
            for col in existing_categorical:
                if col in self.target_encoded_features:
                    X[col] = self._apply_target_encoding(X[col], col)
                else:
                    # Label encoding для низько-кардинальних
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        # Заповнюємо пропуски перед енкодингом
                        X[col] = X[col].fillna('missing')
                        self.label_encoders[col].fit(X[col])
                    
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
        df_temp = pd.DataFrame({'category': series, 'target': target})
        stats = df_temp.groupby('category')['target'].agg(['sum', 'count']).reset_index()
        
        # Згладжування (smoothing)
        smoothing = 10
        stats['smoothed_mean'] = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        # Створюємо словник для швидкого доступу
        encoding_dict = dict(zip(stats['category'], stats['smoothed_mean']))
        
        self.target_encoding_stats[col_name] = {
            'encoding_dict': encoding_dict,
            'global_mean': global_mean
        }
    
    def _apply_target_encoding(self, series: pd.Series, col_name: str) -> pd.Series:
        """Застосування target encoding"""
        if col_name not in self.target_encoding_stats:
            # Якщо немає статистики, повертаємо 0
            return pd.Series([0] * len(series), index=series.index)
        
        encoding_dict = self.target_encoding_stats[col_name]['encoding_dict']
        global_mean = self.target_encoding_stats[col_name]['global_mean']
        
        # Мапінг значень
        encoded = series.map(encoding_dict)
        
        # Заповнення пропусків глобальним середнім
        encoded = encoded.fillna(global_mean)
        
        return encoded
    
    def _safe_label_encode(self, value, col_name: str):
        """Безпечне label encoding з обробкою невідомих значень"""
        if col_name not in self.label_encoders:
            return -1
            
        try:
            # Перевіряємо чи значення було в навчальній вибірці
            if value in self.label_encoders[col_name].classes_:
                return self.label_encoders[col_name].transform([value])[0]
            else:
                return -1  # Невідома категорія
        except:
            return -1