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
        self.market_stats = None  # Буде ініціалізовано пізніше
        
        # Завантаження брендів з файлу
        try:
            with open(brands_file, 'r', encoding='utf-8') as f:
                brands_data = json.load(f)
                self.brand_patterns = brands_data.get('brands', [])
        except Exception as e:
            self.logger.warning(f"Не вдалося завантажити brands.json: {e}")
            self.brand_patterns = []
    
    def set_market_statistics(self, market_stats):
        """Встановлення об'єкту ринкової статистики"""
        self.market_stats = market_stats
        
    def extract_features(self, item: Dict, supplier_profile: Optional[Dict] = None) -> Dict:
        """Витягування всіх ознак для одного запису"""
        features = {}
        
        # Отримуємо категорію тендера
        tender_category = item.get('F_INDUSTRYNAME', 'unknown')
        
        # 1. Категоріальні ознаки (головний фокус)
        # Отримуємо primary_category з профілю постачальника
        if supplier_profile and 'categories' in supplier_profile:
            categories = supplier_profile.get('categories', {})
            if categories:
                # Знаходимо категорію з найбільшим 'won'
                primary_category = max(
                    categories.items(),
                    key=lambda x: x[1].get('won', 0)
                )[0]
                features['primary_category'] = primary_category
            else:
                features['primary_category'] = 'unknown'
        else:
            features['primary_category'] = 'unknown'
        
        # 2. Ринкова статистика для категорії
        if self.market_stats:
            category_context = self.market_stats.get_category_context(tender_category)
            
            # Базові метрики категорії
            features['category_avg_suppliers'] = category_context['avg_suppliers_per_tender']
            features['category_win_probability'] = category_context['empirical_win_probability']
            features['category_market_openness'] = category_context['market_openness']
            features['category_entry_barrier'] = category_context['entry_barrier_score']
            
            # Спеціальні метрики для нових постачальників
            if not supplier_profile or supplier_profile.get('metrics', {}).get('total_tenders', 0) < 5:
                features['is_new_supplier'] = 1
                features['category_new_supplier_win_rate'] = category_context['new_supplier_win_rate']
            else:
                features['is_new_supplier'] = 0
                features['category_new_supplier_win_rate'] = 0
            
            # Індикатори складності ринку
            features['is_high_competition'] = 1 if category_context['avg_suppliers_per_tender'] > 5 else 0
            features['is_concentrated_market'] = 1 if category_context.get('market_concentration', 0) > 0.3 else 0
            features['is_open_market'] = 1 if category_context['market_openness'] > 0.7 else 0
        else:
            # Дефолтні значення якщо статистика недоступна
            features['category_avg_suppliers'] = 3.0
            features['category_win_probability'] = 0.33
            features['category_market_openness'] = 0.7
            features['category_entry_barrier'] = 0.5
            features['is_new_supplier'] = 0
            features['category_new_supplier_win_rate'] = 0.2
            features['is_high_competition'] = 0
            features['is_concentrated_market'] = 0
            features['is_open_market'] = 1
        
        # 3. Ознаки постачальника (тільки профільні)
        if supplier_profile:
            # Метрики постачальника
            metrics = supplier_profile.get('metrics', {})
            features['supplier_win_rate'] = metrics.get('win_rate', 0)
            features['supplier_position_win_rate'] = metrics.get('position_win_rate', 0)
            features['supplier_experience'] = metrics.get('total_tenders', 0)
            features['supplier_stability'] = metrics.get('stability_score', 0)
            features['supplier_specialization'] = metrics.get('specialization_score', 0)
            features['supplier_recent_win_rate'] = metrics.get('recent_win_rate', 0)
            features['supplier_growth_rate'] = metrics.get('growth_rate', 0)
            features['supplier_reliability'] = supplier_profile.get('reliability_score', 0)
            
            # Категоріальна експертиза постачальника в поточній категорії тендера
            cat_data = supplier_profile.get('categories', {}).get(tender_category, {})
            features['supplier_category_experience'] = cat_data.get('total', 0)
            features['supplier_category_win_rate'] = cat_data.get('win_rate', 0)
            
            # Відносні метрики (постачальник vs ринок)
            if self.market_stats and features['supplier_win_rate'] > 0:
                features['supplier_vs_market_avg'] = features['supplier_win_rate'] - features['category_win_probability']
                features['supplier_percentile'] = self.market_stats.calculate_supplier_percentile(
                    supplier_profile, tender_category
                )
            else:
                features['supplier_vs_market_avg'] = 0
                features['supplier_percentile'] = 0.5
            features['supplier_category_win_rate'] = cat_data.get('win_rate', 0)
            
            # Кластерні ознаки (завжди розраховуються якщо є профіль)
            supplier_clusters = supplier_profile.get('clusters', [])
            features['supplier_cluster_count'] = len(supplier_clusters)
            
            # Конкурентні ознаки на основі кластерів
            top_competitors = supplier_profile.get('top_competitors', [])
            bottom_competitors = supplier_profile.get('bottom_competitors', [])
            
            if top_competitors:
                avg_top_win_rate = sum(c['year_win_rate'] for c in top_competitors) / len(top_competitors)
                features['competitor_top_avg_win_rate'] = avg_top_win_rate
                
                # Порівняння з топ конкурентами
                supplier_win_rate = metrics.get('win_rate', 0)
                features['supplier_vs_top_competitors'] = supplier_win_rate - avg_top_win_rate
                
                # Кількість сильних конкурентів
                features['strong_competitors_count'] = len([c for c in top_competitors if c['year_win_rate'] > 0.5])
            else:
                features['competitor_top_avg_win_rate'] = 0
                features['supplier_vs_top_competitors'] = 0
                features['strong_competitors_count'] = 0
            
            # Чи є лідером в своїх кластерах
            total_wins = metrics.get('won_positions', 0)
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
            # Дефолтні значення для нового постачальника
            # Використовуємо ринкову статистику замість нулів
            if self.market_stats:
                context = self.market_stats.get_category_context(tender_category)
                features['supplier_win_rate'] = context['new_supplier_win_rate']
                features['supplier_position_win_rate'] = context['new_supplier_win_rate']
                features['supplier_experience'] = 0  # Це важливий індикатор новачка
                features['supplier_stability'] = 0.3  # Середня стабільність
                features['supplier_specialization'] = 0.5  # Середня спеціалізація
                features['supplier_recent_win_rate'] = context['new_supplier_win_rate']
                features['supplier_growth_rate'] = 0.0
                features['supplier_reliability'] = 0.3
                features['supplier_category_experience'] = 0
                features['supplier_category_win_rate'] = context['new_supplier_win_rate']
                features['supplier_vs_market_avg'] = 0
                features['supplier_percentile'] = 0.25  # Новачок в нижньому квартилі
            else:
                # Фолбек на старі нулі якщо немає статистики
                for key in ['supplier_win_rate', 'supplier_position_win_rate', 'supplier_experience',
                           'supplier_stability', 'supplier_specialization', 'supplier_recent_win_rate',
                           'supplier_growth_rate', 'supplier_reliability', 'supplier_category_experience',
                           'supplier_category_win_rate', 'supplier_vs_market_avg', 'supplier_percentile']:
                    features[key] = 0.0
            
            # Кластерні ознаки для нового постачальника
            features['supplier_cluster_count'] = 0
            features['competitor_top_avg_win_rate'] = 0
            features['supplier_vs_top_competitors'] = 0
            features['strong_competitors_count'] = 0
            features['is_cluster_leader'] = 0
            
            # Дефолтні значення для кластерів
            for cluster in ['agricultural_parts', 'construction', 'medical', 
                        'office_supplies', 'electronics', 'food', 'fuel', 
                        'services', 'communication']:
                features[f'works_in_{cluster}'] = 0
            supplier_clusters = supplier_profile.get('clusters', [])
            features['supplier_cluster_count'] = len(supplier_clusters)
            
            # Конкурентні ознаки на основі кластерів
            top_competitors = supplier_profile.get('top_competitors', [])
            bottom_competitors = supplier_profile.get('bottom_competitors', [])
            
            if top_competitors:
                avg_top_win_rate = sum(c['year_win_rate'] for c in top_competitors) / len(top_competitors)
                features['competitor_top_avg_win_rate'] = avg_top_win_rate
                
                # Порівняння з топ конкурентами
                supplier_win_rate = metrics.get('win_rate', 0)
                features['supplier_vs_top_competitors'] = supplier_win_rate - avg_top_win_rate
                
                # Кількість сильних конкурентів
                features['strong_competitors_count'] = len([c for c in top_competitors if c['year_win_rate'] > 0.5])
            else:
                features['competitor_top_avg_win_rate'] = 0
                features['supplier_vs_top_competitors'] = 0
                features['strong_competitors_count'] = 0
            
            # Чи є лідером в своїх кластерах
            total_wins = metrics.get('won_positions', 0)
            features['is_cluster_leader'] = 1 if total_wins > 500 else 0
            
            # Специфічні кластери як бінарні ознаки
            important_clusters = [
                'agricultural_parts', 'construction', 'medical', 
                'office_supplies', 'electronics', 'food', 'fuel', 
                'services', 'communication'
            ]
            
            for cluster in important_clusters:
                features[f'works_in_{cluster}'] = 1 if cluster in supplier_clusters else 0
       
        
        # 3. Текстові ознаки з назви позиції
        item_name = item.get('F_ITEMNAME', '')
        features['item_name_length'] = len(item_name)
        features['item_name_words'] = len(item_name.split())
        
        # Наявність брендів
        item_upper = item_name.upper()
        features['has_brand'] = int(any(brand in item_upper for brand in self.brand_patterns))
        features['brand_count'] = sum(1 for brand in self.brand_patterns if brand in item_upper)
        
        # 4. CPV ознаки (якщо є)
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
        
        # 6. Складені ознаки (без фінансових)
        features['supplier_category_fit'] = features['supplier_category_win_rate'] * features['supplier_specialization']
        features['competitive_strength'] = features['supplier_win_rate'] * features['supplier_stability']
        features['market_position_score'] = features['supplier_experience'] * features['supplier_win_rate'] / 100
        
        # Нові складені ознаки з урахуванням ринку
        if self.market_stats:
            # Відносна сила в категорії
            features['relative_category_strength'] = (
                features['supplier_category_win_rate'] / 
                (features['category_win_probability'] + 0.01)  # уникаємо ділення на 0
            )
            
            # Адаптованість до конкуренції
            features['competition_readiness'] = (
                features['supplier_experience'] * 
                (1 - features['category_entry_barrier']) *
                features['supplier_stability']
            ) / 100
        
        # Оновлення списку ознак
        if not self.feature_names:
            self.feature_names = list(features.keys())
        
        return features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Створення взаємодіючих ознак"""
        # Копія для уникнення змін оригіналу
        df = df.copy()
        
        # Поліноміальні ознаки для ключових метрик
        key_features = ['supplier_experience', 'supplier_win_rate', 'supplier_category_win_rate']
        
        # Взаємодії другого порядку
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        
        # Квадрати ключових ознак
        for feat in key_features:
            if feat in df.columns:
                df[f'{feat}_squared'] = df[feat] ** 2
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Отримання списку всіх ознак"""
        return self.feature_names