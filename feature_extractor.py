# feature_extractor.py
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import re
import logging
import json


class FeatureExtractor:
    """Клас для вилучення ознак з даних тендерів"""
    
    def __init__(self, categories_manager, competition_analyzer, brands_file='data/brands.json'):
        self.category_manager = categories_manager
        self.competition_analyzer = competition_analyzer
        self.logger = logging.getLogger(__name__)
        self.feature_names = []
        
        # Завантаження брендів з файлу
        try:
            with open(brands_file, 'r', encoding='utf-8') as f:
                brands_data = json.load(f)
                self.brand_patterns = brands_data.get('brands', [])
        except Exception as e:
            self.logger.warning(f"Не вдалося завантажити brands.json: {e}")
            self.brand_patterns = []
        
    def extract_features(self, item: Dict, supplier_profile: Optional[Dict] = None) -> Dict:
        """Витягування всіх ознак для одного запису"""
        features = {}
        
        # 1. Базові числові ознаки
        # features['budget'] = float(item.get('ITEM_BUDGET', 0))
        # features['quantity'] = float(item.get('F_qty', 0))
        # features['price'] = float(item.get('F_price', 0))
        # features['budget_per_unit'] = features['budget'] / (features['quantity'] + 1)
        
        # 2. Категоріальні ознаки
        # Якщо у профілі є категорії — беремо ту, де найбільше виграних позицій
        if supplier_profile and hasattr(supplier_profile, 'categories') and supplier_profile.categories:
            # Знаходимо категорію з найбільшим 'won'
            primary_category = max(
                supplier_profile.categories.items(),
                key=lambda x: x[1].get('won', 0)
            )[0]
            features['primary_category'] = primary_category
        else:
            features['primary_category'] = 'unknown'        
        
        # 3. Ознаки постачальника
        if supplier_profile:
            metrics = getattr(supplier_profile, 'metrics', {})
            features['supplier_win_rate'] = getattr(metrics, 'win_rate', 0)
            features['supplier_position_win_rate'] = getattr(metrics, 'position_win_rate', 0)
            features['supplier_experience'] = getattr(metrics, 'total_tenders', 0)
            features['supplier_stability'] = getattr(metrics, 'stability_score', 0)
            features['supplier_specialization'] = getattr(metrics, 'specialization_score', 0)
            features['supplier_recent_win_rate'] = getattr(metrics, 'recent_win_rate', 0)
            features['supplier_growth_rate'] = getattr(metrics, 'growth_rate', 0)
            features['supplier_reliability'] = getattr(supplier_profile, 'reliability_score', 0)
            
            # Категоріальна експертиза постачальника
            cat_data = getattr(supplier_profile, 'categories', {}).get(features['primary_category'], {})
            features['supplier_category_experience'] = cat_data.get('total', 0)
            features['supplier_category_win_rate'] = cat_data.get('win_rate', 0)
        else:
            # Дефолтні значення
            for key in ['supplier_win_rate', 'supplier_position_win_rate', 'supplier_experience',
                       'supplier_stability', 'supplier_specialization', 'supplier_recent_win_rate',
                       'supplier_growth_rate', 'supplier_reliability', 'supplier_category_experience',
                       'supplier_category_win_rate']:
                features[key] = 0.0
        
        # # 4. Конкурентні ознаки
        # try:
        #     competition_metrics = self.competition_analyzer.calculate_competition_metrics(
        #         features['primary_category']
        #     )
        #     features['competition_intensity'] = competition_metrics.intensity
        #     features['market_concentration'] = competition_metrics.market_concentration
        #     features['entry_barrier'] = competition_metrics.entry_barrier
        #     features['price_volatility'] = competition_metrics.price_volatility
        #     features['avg_participants'] = competition_metrics.avg_participants
        #     features['market_stability'] = competition_metrics.market_stability
        # except:
        #     # Дефолтні значення при помилці
        #     for key in ['competition_intensity', 'market_concentration', 'entry_barrier',
        #                'price_volatility', 'avg_participants', 'market_stability']:
        #         features[key] = 0.0


        if supplier_profile:
            # Кластери постачальника
            if hasattr(supplier_profile, 'clusters'):
                supplier_clusters = supplier_profile.clusters
            elif isinstance(supplier_profile, dict):
                supplier_clusters = supplier_profile.get('clusters', [])
            else:
                supplier_clusters = []
            features['supplier_cluster_count'] = len(supplier_clusters)
            
            # Ознаки конкуренції на основі кластерів
            if hasattr(supplier_profile, 'top_competitors'):
                top_competitors = supplier_profile.top_competitors
            elif isinstance(supplier_profile, dict):
                top_competitors = supplier_profile.get('top_competitors', [])
            else:
                top_competitors = []

            if hasattr(supplier_profile, 'bottom_competitors'):
                bottom_competitors = supplier_profile.bottom_competitors
            elif isinstance(supplier_profile, dict):
                bottom_competitors = supplier_profile.get('bottom_competitors', [])
            else:
                bottom_competitors = []

            # Метрики топ конкурентів
            if top_competitors:
                avg_top_win_rate = sum(c['year_win_rate'] for c in top_competitors) / len(top_competitors)
                features['competitor_top_avg_win_rate'] = avg_top_win_rate

                # Порівняння з топ конкурентами
                if hasattr(supplier_profile, 'metrics'):
                    supplier_win_rate = getattr(supplier_profile.metrics, 'win_rate', 0)
                elif isinstance(supplier_profile, dict):
                    supplier_win_rate = supplier_profile.get('metrics', {}).get('win_rate', 0)
                else:
                    supplier_win_rate = 0
                features['supplier_vs_top_competitors'] = supplier_win_rate - avg_top_win_rate
                
                # Кількість сильних конкурентів
                features['strong_competitors_count'] = len([c for c in top_competitors if c['year_win_rate'] > 0.5])
            else:
                features['competitor_top_avg_win_rate'] = 0
                features['supplier_vs_top_competitors'] = 0
                features['strong_competitors_count'] = 0
            
            # Чи є лідером в своїх кластерах
            total_wins = 0
            if hasattr(supplier_profile, 'metrics'):
                total_wins = getattr(supplier_profile.metrics, 'won_positions', 0)
            elif isinstance(supplier_profile, dict):
                total_wins = supplier_profile.get('metrics', {}).get('won_positions', 0)
            else:
                total_wins = 0
            features['is_cluster_leader'] = 1 if total_wins > 500 else 0
            
            # Специфічні кластери як бінарні ознаки
            important_clusters = [
                'agricultural_parts', 'construction', 'medical', 
                'office_supplies', 'electronics', 'food', 'fuel', 
                'services', 'communication'
            ]
            
            for cluster in important_clusters:
                features[f'works_in_{cluster}'] = 1 if cluster in supplier_clusters else 0
        else:
            # Дефолтні значення для кластерних ознак
            features['supplier_cluster_count'] = -1
            features['competitor_top_avg_win_rate'] = -1
            features['supplier_vs_top_competitors'] = -1
            features['strong_competitors_count'] = -1
            features['is_cluster_leader'] = -1
        
        # Дефолтні значення для кластерів
        for cluster in ['agricultural_parts', 'construction', 'medical', 
                    'office_supplies', 'electronics', 'food', 'fuel', 
                    'services', 'communication']:
            features[f'works_in_{cluster}'] = -1

        # 5. Темпоральні ознаки
        date_str = item.get('DATEEND') or item.get('date_end')
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
        key_features = ['supplier_experience', 'competition_intensity', 'supplier_win_rate']
        
        # Взаємодії другого порядку
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        
        # Квадрати ключових ознак
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_squared'] = df[feat] ** 2
        
        # # Логарифмічні трансформації для монетарних значень
        # for feat in ['price', 'budget_per_unit']:
        #     if feat in df.columns:
        #         df[f'log_{feat}'] = np.log1p(df[feat])
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Отримання списку всіх ознак"""
        return self.feature_names