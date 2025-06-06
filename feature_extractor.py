# feature_extractor.py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import re
import logging


class FeatureExtractor:
    """Клас для вилучення ознак з даних тендерів"""
    
    def __init__(self, categories_manager, competition_analyzer):
        self.category_manager = categories_manager
        self.competition_analyzer = competition_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Список всіх можливих ознак
        self.feature_names = []
        
        # Патерни брендів
        self.brand_patterns = [
            'FENDT', 'JOHN DEERE', 'CASE', 'NEW HOLLAND', 'CLAAS',
            'CATERPILLAR', 'KOMATSU', 'VOLVO', 'SCANIA', 'MAN'
        ]
        
    def extract_features(self, item: Dict, supplier_profile: Optional[Dict] = None) -> Dict:
        """Витягування всіх ознак для одного запису"""
        features = {}
        
        # 1. Базові числові ознаки
        features['budget'] = float(item.get('ITEM_BUDGET', 0))
        features['quantity'] = float(item.get('F_qty', 0))
        features['price'] = float(item.get('F_price', 0))
        features['budget_per_unit'] = features['budget'] / (features['quantity'] + 1)
        
        # 2. Категоріальні ознаки
        categories = self.category_manager.categorize_item(item.get('F_ITEMNAME', ''))
        features['primary_category'] = categories[0][0] if categories else 'unknown'
        features['category_confidence'] = categories[0][1] if categories else 0.0
        features['num_categories'] = len(categories)
        
        # 3. Ознаки постачальника
        if supplier_profile:
            metrics = supplier_profile.get('metrics', {})
            features['supplier_win_rate'] = metrics.get('win_rate', 0)
            features['supplier_position_win_rate'] = metrics.get('position_win_rate', 0)
            features['supplier_experience'] = metrics.get('total_tenders', 0)
            features['supplier_stability'] = metrics.get('stability_score', 0)
            features['supplier_specialization'] = metrics.get('specialization_score', 0)
            features['supplier_recent_win_rate'] = metrics.get('recent_win_rate', 0)
            features['supplier_growth_rate'] = metrics.get('growth_rate', 0)
            features['supplier_reliability'] = supplier_profile.get('reliability_score', 0)
            
            # Категоріальна експертиза постачальника
            cat_data = supplier_profile.get('categories', {}).get(features['primary_category'], {})
            features['supplier_category_experience'] = cat_data.get('total', 0)
            features['supplier_category_win_rate'] = cat_data.get('win_rate', 0)
        else:
            # Дефолтні значення
            for key in ['supplier_win_rate', 'supplier_position_win_rate', 'supplier_experience',
                       'supplier_stability', 'supplier_specialization', 'supplier_recent_win_rate',
                       'supplier_growth_rate', 'supplier_reliability', 'supplier_category_experience',
                       'supplier_category_win_rate']:
                features[key] = 0.0
        
        # 4. Конкурентні ознаки
        try:
            competition_metrics = self.competition_analyzer.calculate_competition_metrics(
                features['primary_category']
            )
            features['competition_intensity'] = competition_metrics.intensity
            features['market_concentration'] = competition_metrics.market_concentration
            features['entry_barrier'] = competition_metrics.entry_barrier
            features['price_volatility'] = competition_metrics.price_volatility
            features['avg_participants'] = competition_metrics.avg_participants
            features['market_stability'] = competition_metrics.market_stability
        except:
            # Дефолтні значення при помилці
            for key in ['competition_intensity', 'market_concentration', 'entry_barrier',
                       'price_volatility', 'avg_participants', 'market_stability']:
                features[key] = 0.0
        
        # 5. Темпоральні ознаки
        date_str = item.get('DATEEND', '')
        if date_str:
            try:
                date = datetime.strptime(date_str, "%d.%m.%Y")
                features['month'] = date.month
                features['quarter'] = (date.month - 1) // 3 + 1
                features['year'] = date.year
                features['day_of_week'] = date.weekday()
                features['is_month_end'] = 1 if date.day > 25 else 0
                features['is_quarter_end'] = 1 if date.month in [3, 6, 9, 12] and date.day > 20 else 0
            except:
                # Дефолтні значення
                for key in ['month', 'quarter', 'year', 'day_of_week', 'is_month_end', 'is_quarter_end']:
                    features[key] = 0
        
        # 6. Текстові ознаки
        item_name = item.get('F_ITEMNAME', '')
        features['item_name_length'] = len(item_name)
        features['item_name_words'] = len(item_name.split())
        
        # Наявність брендів
        item_upper = item_name.upper()
        features['has_brand'] = int(any(brand in item_upper for brand in self.brand_patterns))
        features['brand_count'] = sum(1 for brand in self.brand_patterns if brand in item_upper)
        
        # Індикатори якості
        quality_indicators = {
            'premium': ['оригінал', 'преміум', 'високоякіс', 'сертифікован'],
            'standard': ['стандарт', 'якісн', 'надійн'],
            'budget': ['економ', 'бюджет', 'аналог']
        }
        
        item_lower = item_name.lower()
        features['is_premium'] = int(any(ind in item_lower for ind in quality_indicators['premium']))
        features['is_standard'] = int(any(ind in item_lower for ind in quality_indicators['standard']))
        features['is_budget'] = int(any(ind in item_lower for ind in quality_indicators['budget']))
        
        # 7. CPV ознаки
        cpv = item.get('CPV', 0)
        if cpv:
            cpv_str = str(cpv)
            features['cpv_division'] = int(cpv_str[:2]) if len(cpv_str) >= 2 else 0
            features['cpv_group'] = int(cpv_str[:3]) if len(cpv_str) >= 3 else 0
            features['cpv_class'] = int(cpv_str[:4]) if len(cpv_str) >= 4 else 0
        else:
            features['cpv_division'] = 0
            features['cpv_group'] = 0
            features['cpv_class'] = 0
        
        # 8. Складені ознаки
        features['supplier_category_fit'] = features['supplier_category_win_rate'] * (1 - features['competition_intensity'])
        features['risk_score'] = features['entry_barrier'] * features['competition_intensity'] * (1 - features['supplier_stability'])
        features['opportunity_score'] = features['supplier_win_rate'] * features['market_stability'] / (features['competition_intensity'] + 0.1)
        
        # Оновлення списку ознак
        if not self.feature_names:
            self.feature_names = list(features.keys())
        
        return features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Створення взаємодіючих ознак"""
        # Копія для уникнення змін оригіналу
        df = df.copy()
        
        # Поліноміальні ознаки для ключових метрик
        key_features = ['budget', 'supplier_experience', 'competition_intensity', 'supplier_win_rate']
        
        # Взаємодії другого порядку
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        
        # Квадрати ключових ознак
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_squared'] = df[feat] ** 2
        
        # Логарифмічні трансформації для монетарних значень
        for feat in ['budget', 'price', 'budget_per_unit']:
            if feat in df.columns:
                df[f'log_{feat}'] = np.log1p(df[feat])
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Отримання списку всіх ознак"""
        return self.feature_names