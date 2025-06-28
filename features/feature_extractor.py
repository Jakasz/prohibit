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
    
    def __init__(self, categories_manager,  brands_file='data/brands.json'):
        self.categories_manager = categories_manager        
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
        
    def extract_features(self, item: Dict, supplier_profile: Optional[Any] = None) -> Dict:
        """Витягування всіх ознак для одного запису"""
        features = {}
        tender_category = item.get('F_INDUSTRYNAME', 'unknown')

        # 1. Визначаємо чи є досвід в категорії
        has_category_experience = False
        category_experience_data = None
        
        if supplier_profile:
            categories = None
            if hasattr(supplier_profile, 'categories'):
                categories = supplier_profile.categories
            elif isinstance(supplier_profile, dict):
                categories = supplier_profile.get('categories', {})
            
            if categories and tender_category in categories:
                category_experience_data = categories[tender_category]
                if category_experience_data.get('total', 0) >= 5:  # Мінімум 5 тендерів для "досвіду"
                    has_category_experience = True
        
        features['has_category_experience'] = 1 if has_category_experience else 0
        
        # 2. Кластерна логіка
        cluster_name = None
        if self.market_stats and hasattr(self.market_stats, 'get_category_cluster'):
            cluster_name = self.market_stats.get_category_cluster(tender_category)

        # 3. Визначаємо тип досвіду та заповнюємо метрики
        if has_category_experience:
            # Прямий досвід в категорії
            features['experience_type'] = 1  # category
            features['supplier_category_experience'] = category_experience_data.get('total', 0)
            features['supplier_category_win_rate'] = category_experience_data.get('win_rate', 0)
            features['supplier_category_wins'] = category_experience_data.get('won', 0)
            
        elif cluster_name and supplier_profile and categories:
            # Досвід в кластері
            features['experience_type'] = 2  # cluster
            
            # Збираємо статистику по всіх категоріях кластера
            cluster_total = 0
            cluster_won = 0
            cluster_categories_count = 0

            cluster_categories = self.categories_manager.categories_map.get(cluster_name, [])

            for cat in cluster_categories:
                if cat in categories and cat != tender_category:
                    cat_data = categories[cat]
                    cluster_total += cat_data.get('total', 0)
                    cluster_won += cat_data.get('won', 0)
                    if cat_data.get('total', 0) > 0:
                        cluster_categories_count += 1
            
            # Кластерні метрики
            cluster_win_rate = cluster_won / cluster_total if cluster_total > 0 else 0
            
            # Використовуємо кластерні дані з понижуючим коефіцієнтом
            features['supplier_category_experience'] = cluster_total * 0.7
            features['supplier_category_win_rate'] = cluster_win_rate * 0.8
            features['supplier_category_wins'] = cluster_won * 0.7
            
        else:
            # Немає релевантного досвіду
            features['experience_type'] = 3  # general
            features['supplier_category_experience'] = 0
            if isinstance(supplier_profile, dict):
                features['supplier_category_win_rate'] = supplier_profile.get('metrics', {}).get('win_rate', 0) * 0.5
            elif supplier_profile:
                metrics = getattr(supplier_profile, 'metrics', {})
                features['supplier_category_win_rate'] = getattr(metrics, 'win_rate', 0) * 0.5
            else:
                features['supplier_category_win_rate'] = 0
                features['supplier_category_wins'] = 0
        
        # 4. Основні метрики постачальника
        if supplier_profile:
            if isinstance(supplier_profile, dict):
                metrics = supplier_profile.get('metrics', {})
                features['supplier_win_rate'] = metrics.get('win_rate', 0.0)
                features['supplier_position_win_rate'] = metrics.get('position_win_rate', 0.0)
                features['supplier_experience'] = metrics.get('total_positions', 0)
                features['supplier_stability'] = metrics.get('stability_score', 0.0)
                features['supplier_specialization'] = metrics.get('specialization_score', 0.0)
                features['supplier_recent_win_rate'] = metrics.get('recent_win_rate', 0.0)
                features['supplier_growth_rate'] = metrics.get('growth_rate', 0.0)
                features['supplier_reliability'] = supplier_profile.get('reliability_score', 0.0)
            else:
                metrics = getattr(supplier_profile, 'metrics', {})
                features['supplier_win_rate'] = getattr(metrics, 'win_rate', 0.0)
                features['supplier_position_win_rate'] = getattr(metrics, 'position_win_rate', 0.0)
                features['supplier_experience'] = getattr(metrics, 'total_positions', 0)
                features['supplier_stability'] = getattr(metrics, 'stability_score', 0.0)
                features['supplier_specialization'] = getattr(metrics, 'specialization_score', 0.0)
                features['supplier_recent_win_rate'] = getattr(metrics, 'recent_win_rate', 0.0)
                features['supplier_growth_rate'] = getattr(metrics, 'growth_rate', 0.0)
                features['supplier_reliability'] = getattr(supplier_profile, 'reliability_score', 0.0)

            # # Кластерні ознаки
            # if hasattr(supplier_profile, 'clusters'):
            #     supplier_clusters = supplier_profile.clusters
            #     features['supplier_cluster_count'] = len(supplier_clusters)
            #     features['works_in_current_cluster'] = 1 if cluster_name and cluster_name in supplier_clusters else 0
            # else:
            #     features['supplier_cluster_count'] = 0
            #     features['works_in_current_cluster'] = 0
                
            # Конкурентні ознаки
            if hasattr(supplier_profile, 'top_competitors'):
                top_competitors = supplier_profile.top_competitors
                if top_competitors:
                    avg_top_win_rate = sum(c.get('year_win_rate', 0) for c in top_competitors) / len(top_competitors)
                    features['competitor_top_avg_win_rate'] = avg_top_win_rate
                    features['supplier_vs_top_competitors'] = metrics.win_rate - avg_top_win_rate
                else:
                    features['competitor_top_avg_win_rate'] = 0
                    features['supplier_vs_top_competitors'] = 0
            else:
                features['competitor_top_avg_win_rate'] = 0
                features['supplier_vs_top_competitors'] = 0
                
        else:
            # Дефолтні значення для нового постачальника
            for key in ['supplier_win_rate', 'supplier_position_win_rate', 'supplier_experience',
                    'supplier_stability', 'supplier_specialization', 'supplier_recent_win_rate',
                    'supplier_growth_rate', 'supplier_reliability']:
                features[key] = 0.0
            # features['supplier_cluster_count'] = 0
            # features['works_in_current_cluster'] = 0
            features['competitor_top_avg_win_rate'] = 0
            features['supplier_vs_top_competitors'] = 0
        
        # 5. Ринкова статистика для категорії
        if self.market_stats:
            category_context = self.market_stats.get_category_context(tender_category)
            features['category_avg_suppliers'] = category_context['avg_suppliers_per_tender']
            features['category_win_probability'] = category_context['empirical_win_probability']
            features['category_market_openness'] = category_context['market_openness']
            features['category_entry_barrier'] = category_context['entry_barrier_score']
            
            # Визначаємо чи новий постачальник
            if not supplier_profile:
                features['is_new_supplier'] = 1
            elif isinstance(supplier_profile, dict):
                features['is_new_supplier'] = 1 if supplier_profile.get('metrics', {}).get('total_tenders', 0) < 5 else 0
            else:
                metrics = getattr(supplier_profile, 'metrics', {})
                features['is_new_supplier'] = 1 if getattr(metrics, 'total_tenders', 0) < 5 else 0

            # Відносна сила постачальника
            if supplier_profile and features['supplier_win_rate'] > 0:
                features['supplier_vs_market_avg'] = features['supplier_win_rate'] - features['category_win_probability']
            else:
                features['supplier_vs_market_avg'] = 0
        else:
            # Дефолтні значення
            features['category_avg_suppliers'] = 3.0
            features['category_win_probability'] = 0.33
            features['category_market_openness'] = 0.7
            features['category_entry_barrier'] = 0.5
            features['is_new_supplier'] = 1 if not supplier_profile else 0
            features['supplier_vs_market_avg'] = 0
        
        # 6. Текстові ознаки (тільки базові)
        # item_name = item.get('F_ITEMNAME', '')
        # features['item_name_length'] = len(item_name)
        # features['item_name_words'] = len(item_name.split())
        
        # Наявність брендів
        # item_upper = item_name.upper()
        # features['has_brand'] = int(any(brand in item_upper for brand in self.brand_patterns))
        
        # 7. Складені ознаки
        features['supplier_category_fit'] = features['supplier_category_win_rate'] * features['supplier_specialization']
        features['competitive_strength'] = features['supplier_win_rate'] * features['supplier_stability']
        
        # Зберігаємо список ознак
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