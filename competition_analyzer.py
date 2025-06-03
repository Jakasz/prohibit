import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import statistics
from dataclasses import dataclass


@dataclass
class CompetitionMetrics:
    """Структура метрик конкуренції"""
    intensity: float  # 0-1 (низька-висока)
    market_concentration: float  # HHI (0-1)
    entry_barrier: float  # 0-1 (низький-високий)
    price_volatility: float  # Коефіцієнт варіації цін
    avg_participants: float  # Середня кількість учасників
    win_rate_variance: float  # Варіабельність win rate
    market_stability: float  # 0-1 (нестабільний-стабільний)


class CompetitionAnalyzer:
    """
    Аналізатор конкуренції в тендерах
    
    Функції:
    - Аналіз інтенсивності конкуренції по категоріях
    - Розрахунок метрик концентрації ринку
    - Виявлення монополістів та домінуючих гравців
    - Аналіз бар'єрів входу
    - Прогнозування конкурентного тиску
    - Рекомендації для участі в тендерах
    """
    
    def __init__(self, categories_manager, vector_db):
        self.logger = logging.getLogger(__name__)
        self.categories_manager = categories_manager
        self.vector_db = vector_db
        
        # Дані для аналізу конкуренції
        self.competition_data = defaultdict(lambda: {
            'tenders': defaultdict(list),  # tender_id -> [participants]
            'suppliers': defaultdict(lambda: {'participated': 0, 'won': 0, 'total_value': 0}),
            'price_history': [],
            'temporal_data': defaultdict(list),
            'category_crossover': defaultdict(set)  # перехресні категорії постачальників
        })
        
        # Кеш розрахованих метрик
        self.metrics_cache = {}
        self.cache_timestamp = {}
        
        # Порогові значення для класифікації
        self.thresholds = {
            'high_competition': 0.7,
            'medium_competition': 0.4,
            'high_concentration': 0.25,  # HHI
            'dominant_share': 0.4,  # Частка ринку
            'barrier_threshold': 0.6,
            'volatility_threshold': 0.3
        }
        
        self.logger.info("✅ CompetitionAnalyzer ініціалізовано")
    
    def update_competition_metrics(self, historical_data: List[Dict]):
        """Оновлення метрик конкуренції на основі історичних даних"""
        self.logger.info(f"🔄 Оновлення метрик конкуренції для {len(historical_data)} записів...")
        
        # Очищення старих даних
        self.competition_data = defaultdict(lambda: {
            'tenders': defaultdict(list),
            'suppliers': defaultdict(lambda: {'participated': 0, 'won': 0, 'total_value': 0}),
            'price_history': [],
            'temporal_data': defaultdict(list),
            'category_crossover': defaultdict(set)
        })
        
        # Групування даних по категоріях та тендерах
        category_tenders = defaultdict(lambda: defaultdict(list))
        
        for item in historical_data:
            # Визначення категорії
            item_name = item.get('F_ITEMNAME', '')
            categories = self.categories_manager.categorize_item(item_name)
            primary_category = categories[0][0] if categories else 'unknown'
            
            tender_id = item.get('F_TENDERNUMBER', '')
            edrpou = item.get('EDRPOU', '')
            
            if tender_id and edrpou:
                category_tenders[primary_category][tender_id].append(item)
        
        # Аналіз кожної категорії
        for category, tenders in category_tenders.items():
            self._analyze_category_competition_detailed(category, tenders)
        
        # Очищення кешу
        self.metrics_cache.clear()
        self.cache_timestamp.clear()
        
        self.logger.info(f"✅ Оновлено метрики для {len(category_tenders)} категорій")
    
    def _analyze_category_competition_detailed(self, category: str, category_tenders: Dict[str, List[Dict]]):
        """Детальний аналіз конкуренції в категорії"""
        comp_data = self.competition_data[category]
        
        for tender_id, tender_items in category_tenders.items():
            participants = []
            prices = []
            
            for item in tender_items:
                edrpou = item.get('EDRPOU', '')
                if not edrpou:
                    continue
                
                participants.append(edrpou)
                
                # Оновлення даних постачальника
                supplier_data = comp_data['suppliers'][edrpou]
                supplier_data['participated'] += 1
                
                # Ціна
                budget = item.get('ITEM_BUDGET')
                if budget:
                    try:
                        price = float(budget)
                        prices.append(price)
                        if item.get('WON'):
                            supplier_data['total_value'] += price
                    except:
                        pass
                
                # Перемога
                if item.get('WON'):
                    supplier_data['won'] += 1
                
                # Темпоральні дані
                date_end = item.get('DATEEND')
                if date_end:
                    try:
                        date_obj = datetime.strptime(date_end, "%d.%m.%Y")
                        month_key = date_obj.strftime("%Y-%m")
                        comp_data['temporal_data'][month_key].append({
                            'supplier': edrpou,
                            'won': item.get('WON', False),
                            'price': price if 'price' in locals() else 0
                        })
                    except:
                        pass
            
            # Збереження даних тендера
            unique_participants = list(set(participants))
            comp_data['tenders'][tender_id] = unique_participants
            
            # Збереження цін для аналізу волатильності
            if prices:
                comp_data['price_history'].extend(prices)
        
        # Аналіз перехресних категорій (постачальники в декількох категоріях)
        for supplier in comp_data['suppliers'].keys():
            comp_data['category_crossover'][supplier].add(category)
    
    def calculate_competition_metrics(self, category: str) -> CompetitionMetrics:
        """Розрахунок комплексних метрик конкуренції для категорії"""
        # Перевірка кешу
        cache_key = f"{category}_metrics"
        if (cache_key in self.metrics_cache and 
            cache_key in self.cache_timestamp and
            (datetime.now() - self.cache_timestamp[cache_key]).seconds < 3600):  # Кеш на 1 год
            return self.metrics_cache[cache_key]
        
        if category not in self.competition_data:
            # Повертаємо дефолтні метрики для невідомої категорії
            default_metrics = CompetitionMetrics(
                intensity=0.0,
                market_concentration=0.0,
                entry_barrier=0.0,
                price_volatility=0.0,
                avg_participants=0.0,
                win_rate_variance=0.0,
                market_stability=0.0
            )
            self.metrics_cache[cache_key] = default_metrics
            self.cache_timestamp[cache_key] = datetime.now()
            return default_metrics
        
        comp_data = self.competition_data[category]
        
        # 1. Інтенсивність конкуренції
        participants_per_tender = [len(participants) for participants in comp_data['tenders'].values()]
        avg_participants = np.mean(participants_per_tender) if participants_per_tender else 0
        intensity = min(avg_participants / 8, 1.0)  # Нормалізація до 1
        
        # 2. Концентрація ринку (HHI - Herfindahl-Hirschman Index)
        if comp_data['suppliers']:
            total_participations = sum(data['participated'] for data in comp_data['suppliers'].values())
            if total_participations > 0:
                market_shares = [
                    (data['participated'] / total_participations) ** 2 
                    for data in comp_data['suppliers'].values()
                ]
                market_concentration = sum(market_shares)
            else:
                market_concentration = 0.0
        else:
            market_concentration = 0.0
        
        # 3. Бар'єр входу
        # Базується на мінімальній вартості участі та складності
        prices = comp_data['price_history']
        if prices:
            min_price = min(prices)
            avg_price = np.mean(prices)
            # Нормалізація: високі ціни = високий бар'єр
            price_barrier = min(min_price / 100000, 1.0) * 0.4
            # Складність участі (кількість постачальників)
            complexity_barrier = (1 / max(len(comp_data['suppliers']), 1)) * 0.6
            entry_barrier = price_barrier + complexity_barrier
        else:
            entry_barrier = 0.5  # Середній бар'єр за замовчуванням
        
        # 4. Волатильність цін
        price_volatility = 0.0
        if len(prices) > 1:
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            if price_mean > 0:
                price_volatility = price_std / price_mean
        
        # 5. Варіабельність win rate
        win_rates = []
        for supplier_data in comp_data['suppliers'].values():
            if supplier_data['participated'] > 0:
                win_rate = supplier_data['won'] / supplier_data['participated']
                win_rates.append(win_rate)
        
        win_rate_variance = np.var(win_rates) if len(win_rates) > 1 else 0.0
        
        # 6. Стабільність ринку (базується на темпоральних даних)
        market_stability = self._calculate_market_stability(comp_data['temporal_data'])
        
        metrics = CompetitionMetrics(
            intensity=intensity,
            market_concentration=market_concentration,
            entry_barrier=min(entry_barrier, 1.0),
            price_volatility=min(price_volatility, 2.0),  # Обмежуємо екстремальні значення
            avg_participants=avg_participants,
            win_rate_variance=win_rate_variance,
            market_stability=market_stability
        )
        
        # Збереження в кеш
        self.metrics_cache[cache_key] = metrics
        self.cache_timestamp[cache_key] = datetime.now()
        
        return metrics
    
    def _calculate_market_stability(self, temporal_data: Dict[str, List[Dict]]) -> float:
        """Розрахунок стабільності ринку на основі темпоральних даних"""
        if not temporal_data:
            return 0.5  # Нейтральна стабільність
        
        monthly_participants = []
        monthly_winners = []
        
        for month, activities in temporal_data.items():
            participants = set(activity['supplier'] for activity in activities)
            winners = set(activity['supplier'] for activity in activities if activity['won'])
            
            monthly_participants.append(len(participants))
            monthly_winners.append(len(winners))
        
        # Стабільність = низька варіабельність кількості учасників та переможців
        if len(monthly_participants) > 1:
            participants_cv = np.std(monthly_participants) / np.mean(monthly_participants) if np.mean(monthly_participants) > 0 else 1
            winners_cv = np.std(monthly_winners) / np.mean(monthly_winners) if np.mean(monthly_winners) > 0 else 1
            
            # Інвертуємо: низька варіабельність = висока стабільність
            stability = 1 - min((participants_cv + winners_cv) / 2, 1.0)
        else:
            stability = 0.5
        
        return stability
    
    def analyze_tender_competition(self, tender_item: Dict) -> Dict[str, Any]:
        """Аналіз конкуренції для конкретного тендера"""
        item_name = tender_item.get('F_ITEMNAME', '')
        categories = self.categories_manager.categorize_item(item_name)
        primary_category = categories[0][0] if categories else 'unknown'
        
        # Базові метрики категорії
        metrics = self.calculate_competition_metrics(primary_category)
        
        # Пошук схожих тендерів для додаткового контексту
        similar_tenders = self.vector_db.search_similar_tenders(tender_item, limit=20)
        
        # Аналіз схожих тендерів
        similar_analysis = self._analyze_similar_tenders(similar_tenders)
        
        # Прогнозування конкуренції
        competition_forecast = self._forecast_competition_level(tender_item, metrics, similar_analysis)
        
        return {
            'category': primary_category,
            'category_metrics': {
                'competition_intensity': metrics.intensity,
                'market_concentration': metrics.market_concentration,
                'entry_barrier': metrics.entry_barrier,
                'price_volatility': metrics.price_volatility,
                'avg_participants': metrics.avg_participants,
                'market_stability': metrics.market_stability
            },
            'similar_tenders_analysis': similar_analysis,
            'competition_forecast': competition_forecast,
            'recommendations': self._generate_competition_recommendations(metrics, competition_forecast)
        }
    
    def _analyze_similar_tenders(self, similar_tenders: List[Dict]) -> Dict[str, Any]:
        """Аналіз схожих тендерів для прогнозування конкуренції"""
        if not similar_tenders:
            return {
                'avg_similarity': 0.0,
                'common_suppliers': [],
                'avg_budget': 0.0,
                'win_patterns': {},
                'competition_level': 'unknown'
            }
        
        # Аналіз схожості
        similarities = [tender['similarity_score'] for tender in similar_tenders]
        avg_similarity = np.mean(similarities)
        
        # Спільні постачальники
        suppliers = [tender['edrpou'] for tender in similar_tenders if tender['edrpou']]
        supplier_counts = Counter(suppliers)
        common_suppliers = supplier_counts.most_common(5)
        
        # Середній бюджет
        budgets = [tender['budget'] for tender in similar_tenders if tender['budget'] > 0]
        avg_budget = np.mean(budgets) if budgets else 0.0
        
        # Патерни перемог
        win_patterns = {}
        for supplier, count in common_suppliers:
            wins = sum(1 for tender in similar_tenders 
                      if tender['edrpou'] == supplier and tender['won'])
            win_patterns[supplier] = {
                'participations': count,
                'wins': wins,
                'win_rate': wins / count if count > 0 else 0
            }
        
        # Рівень конкуренції
        if len(common_suppliers) > 5 and avg_similarity > 0.8:
            competition_level = 'high'
        elif len(common_suppliers) > 2 and avg_similarity > 0.6:
            competition_level = 'medium'
        else:
            competition_level = 'low'
        
        return {
            'avg_similarity': avg_similarity,
            'common_suppliers': common_suppliers,
            'avg_budget': avg_budget,
            'win_patterns': win_patterns,
            'competition_level': competition_level,
            'total_similar': len(similar_tenders)
        }
    
    def _forecast_competition_level(self, 
                                  tender_item: Dict, 
                                  category_metrics: CompetitionMetrics,
                                  similar_analysis: Dict) -> Dict[str, Any]:
        """Прогнозування рівня конкуренції для тендера"""
        
        # Фактори прогнозування
        factors = {}
        
        # 1. Фактор категорії
        category_factor = category_metrics.intensity * 0.4
        factors['category_intensity'] = category_factor
        
        # 2. Фактор схожих тендерів
        similar_factor = 0.0
        if similar_analysis['competition_level'] == 'high':
            similar_factor = 0.8
        elif similar_analysis['competition_level'] == 'medium':
            similar_factor = 0.5
        else:
            similar_factor = 0.2
        factors['similar_tenders'] = similar_factor * 0.3
        
        # 3. Фактор бюджету
        budget = tender_item.get('ITEM_BUDGET', 0)
        try:
            budget_value = float(budget) if budget else 0
            # Високий бюджет = більше конкуренції
            if budget_value > 1000000:
                budget_factor = 0.9
            elif budget_value > 100000:
                budget_factor = 0.6
            elif budget_value > 10000:
                budget_factor = 0.4
            else:
                budget_factor = 0.2
        except:
            budget_factor = 0.3
        
        factors['budget_attractiveness'] = budget_factor * 0.2
        
        # 4. Фактор бар'єрів входу (інвертований)
        barrier_factor = (1 - category_metrics.entry_barrier) * 0.1
        factors['entry_ease'] = barrier_factor
        
        # Загальний прогноз
        total_forecast = sum(factors.values())
        total_forecast = min(max(total_forecast, 0.0), 1.0)  # Нормалізація 0-1
        
        # Класифікація
        if total_forecast > 0.7:
            competition_level = 'high'
            expected_participants = 5 + int(total_forecast * 10)
        elif total_forecast > 0.4:
            competition_level = 'medium'
            expected_participants = 3 + int(total_forecast * 5)
        else:
            competition_level = 'low'
            expected_participants = 1 + int(total_forecast * 3)
        
        return {
            'competition_score': total_forecast,
            'competition_level': competition_level,
            'expected_participants': min(expected_participants, 20),  # Розумне обмеження
            'factors': factors,
            'confidence': min(similar_analysis.get('avg_similarity', 0) + 0.3, 1.0)
        }
    
    def _generate_competition_recommendations(self, 
                                            metrics: CompetitionMetrics,
                                            forecast: Dict) -> List[str]:
        """Генерація рекомендацій на основі аналізу конкуренції"""
        recommendations = []
        
        competition_level = forecast['competition_level']
        
        if competition_level == 'high':
            recommendations.extend([
                "Очікується висока конкуренція - підготуйте конкурентну пропозицію",
                "Розгляньте можливість диференціації за якістю або додатковими послугами",
                "Ретельно проаналізуйте ціни конкурентів"
            ])
        elif competition_level == 'medium':
            recommendations.extend([
                "Помірна конкуренція - є хороші шанси на перемогу",
                "Зосередьтеся на оптимальному співвідношенні ціна/якість"
            ])
        else:
            recommendations.extend([
                "Низька конкуренція - високі шанси на перемогу",
                "Можливість запропонувати преміум-рішення"
            ])
        
        # Додаткові рекомендації на основі метрик
        if metrics.market_concentration > self.thresholds['high_concentration']:
            recommendations.append("Ринок високо концентрований - аналізуйте стратегії лідерів")
        
        if metrics.price_volatility > self.thresholds['volatility_threshold']:
            recommendations.append("Висока волатільність цін - будьте обережні з ціноутворенням")
        
        if metrics.entry_barrier > self.thresholds['barrier_threshold']:
            recommendations.append("Високі бар'єри входу - переконайтеся у відповідності всім вимогам")
        
        if metrics.market_stability < 0.3:
            recommendations.append("Нестабільний ринок - розгляньте ризики довгострокового планування")
        
        return recommendations
    
    def get_category_competition_metrics(self, category: str) -> Dict[str, Any]:
        """Отримання метрик конкуренції для категорії з інтерпретацією"""
        metrics = self.calculate_competition_metrics(category)
        
        # Інтерпретація метрик
        interpretation = {}
        
        if metrics.intensity < 0.3:
            interpretation['competition'] = "Низька конкуренція"
        elif metrics.intensity < 0.7:
            interpretation['competition'] = "Помірна конкуренція"
        else:
            interpretation['competition'] = "Висока конкуренція"
        
        if metrics.market_concentration < 0.15:
            interpretation['market'] = "Конкурентний ринок"
        elif metrics.market_concentration < 0.25:
            interpretation['market'] = "Помірно концентрований"
        else:
            interpretation['market'] = "Високо концентрований"
        
        # Топ постачальники
        top_suppliers = []
        if category in self.competition_data:
            suppliers = self.competition_data[category]['suppliers']
            sorted_suppliers = sorted(
                suppliers.items(),
                key=lambda x: x[1]['won'],
                reverse=True
            )[:5]
            
            for edrpou, data in sorted_suppliers:
                win_rate = data['won'] / data['participated'] if data['participated'] > 0 else 0
                top_suppliers.append({
                    'edrpou': edrpou,
                    'wins': data['won'],
                    'participations': data['participated'],
                    'win_rate': win_rate,
                    'total_value': data['total_value']
                })
        
        return {
            'metrics': {
                'intensity': metrics.intensity,
                'market_concentration': metrics.market_concentration,
                'entry_barrier': metrics.entry_barrier,
                'price_volatility': metrics.price_volatility,
                'avg_participants': metrics.avg_participants,
                'win_rate_variance': metrics.win_rate_variance,
                'market_stability': metrics.market_stability
            },
            'interpretation': interpretation,
            'top_suppliers': top_suppliers,
            'total_tenders': len(self.competition_data[category]['tenders']) if category in self.competition_data else 0,
            'total_suppliers': len(self.competition_data[category]['suppliers']) if category in self.competition_data else 0
        }
    
    def get_supplier_competition_metrics(self, edrpou: str) -> Dict[str, Any]:
        """Аналіз конкурентної позиції постачальника"""
        supplier_metrics = {
            'categories_active': [],
            'competitive_advantages': [],
            'market_position': {},
            'cross_category_analysis': {}
        }
        
        # Пошук постачальника в різних категоріях
        for category, comp_data in self.competition_data.items():
            if edrpou in comp_data['suppliers']:
                supplier_data = comp_data['suppliers'][edrpou]
                category_metrics = self.calculate_competition_metrics(category)
                
                win_rate = supplier_data['won'] / supplier_data['participated'] if supplier_data['participated'] > 0 else 0
                
                category_info = {
                    'category': category,
                    'participations': supplier_data['participated'],
                    'wins': supplier_data['won'],
                    'win_rate': win_rate,
                    'total_value': supplier_data['total_value'],
                    'category_competition': category_metrics.intensity,
                    'market_share': self._calculate_supplier_market_share(edrpou, category)
                }
                
                supplier_metrics['categories_active'].append(category_info)
        
        # Аналіз конкурентних переваг
        if supplier_metrics['categories_active']:
            avg_win_rate = np.mean([cat['win_rate'] for cat in supplier_metrics['categories_active']])
            total_participations = sum([cat['participations'] for cat in supplier_metrics['categories_active']])
            
            supplier_metrics['market_position'] = {
                'overall_win_rate': avg_win_rate,
                'total_participations': total_participations,
                'active_categories': len(supplier_metrics['categories_active']),
                'specialization_score': self._calculate_specialization_score(supplier_metrics['categories_active'])
            }
            
            # Конкурентні переваги
            if avg_win_rate > 0.6:
                supplier_metrics['competitive_advantages'].append("Високий рівень успішності")
            
            if len(supplier_metrics['categories_active']) > 3:
                supplier_metrics['competitive_advantages'].append("Диверсифікований портфель")
            
            # Аналіз спеціалізації
            specialization = max(supplier_metrics['categories_active'], key=lambda x: x['participations'])
            if specialization['participations'] > total_participations * 0.6:
                supplier_metrics['competitive_advantages'].append(f"Спеціалізація в категорії: {specialization['category']}")
        
        return supplier_metrics
    
    def _calculate_supplier_market_share(self, edrpou: str, category: str) -> float:
        """Розрахунок частки ринку постачальника в категорії"""
        if category not in self.competition_data:
            return 0.0
        
        comp_data = self.competition_data[category]
        if edrpou not in comp_data['suppliers']:
            return 0.0
        
        supplier_participations = comp_data['suppliers'][edrpou]['participated']
        total_participations = sum(data['participated'] for data in comp_data['suppliers'].values())
        
        return supplier_participations / total_participations if total_participations > 0 else 0.0
    
    def _calculate_specialization_score(self, categories_active: List[Dict]) -> float:
        """Розрахунок рівня спеціалізації постачальника"""
        if not categories_active:
            return 0.0
        
        total_participations = sum(cat['participations'] for cat in categories_active)
        if total_participations == 0:
            return 0.0
        
        # Розрахунок індексу концентрації Херфіндаля для участі постачальника
        shares = [(cat['participations'] / total_participations) ** 2 for cat in categories_active]
        hhi = sum(shares)
        
        return hhi
    
    def export_state(self) -> Dict[str, Any]:
        """Експорт стану аналізатора конкуренції"""
        return {
            'competition_data': dict(self.competition_data),
            'metrics_cache': self.metrics_cache,
            'thresholds': self.thresholds
        }
    
    def load_state(self, state_data: Dict[str, Any]):
        """Завантаження стану аналізатора конкуренції"""
        self.competition_data = defaultdict(
            lambda: {
                'tenders': defaultdict(list),
                'suppliers': defaultdict(lambda: {'participated': 0, 'won': 0, 'total_value': 0}),
                'price_history': [],
                'temporal_data': defaultdict(list),
                'category_crossover': defaultdict(set)
            },
            state_data.get('competition_data', {})
        )
        
        self.metrics_cache = state_data.get('metrics_cache', {})
        self.thresholds = state_data.get('thresholds', self.thresholds)
        
        # Очищення timestamp кешу після завантаження
        self.cache_timestamp.clear()
    
    def get_market_leaders(self, category: str, limit: int = 5) -> List[Dict]:
        """Отримання лідерів ринку в категорії"""
        if category not in self.competition_data:
            return []
        
        suppliers = self.competition_data[category]['suppliers']
        
        # Сортування за кількістю перемог та win rate
        leaders = []
        for edrpou, data in suppliers.items():
            win_rate = data['won'] / data['participated'] if data['participated'] > 0 else 0
            market_share = self._calculate_supplier_market_share(edrpou, category)
            
            leaders.append({
                'edrpou': edrpou,
                'wins': data['won'],
                'participations': data['participated'],
                'win_rate': win_rate,
                'market_share': market_share,
                'total_value': data['total_value'],
                'dominance_score': win_rate * 0.6 + market_share * 0.4
            })
        
        # Сортування за dominance_score
        leaders.sort(key=lambda x: x['dominance_score'], reverse=True)
        
        return leaders[:limit]
    
    def detect_market_anomalies(self, category: str) -> List[Dict]:
        """Виявлення ринкових аномалій в категорії"""
        anomalies = []
        
        if category not in self.competition_data:
            return anomalies
        
        metrics = self.calculate_competition_metrics(category)
        comp_data = self.competition_data[category]
        
        # Аномалія: Монополізація
        if metrics.market_concentration > 0.4:
            dominant_supplier = max(
                comp_data['suppliers'].items(),
                key=lambda x: x[1]['participated']
            )
            anomalies.append({
                'type': 'monopolization',
                'severity': 'high' if metrics.market_concentration > 0.6 else 'medium',
                'description': f"Висока концентрація ринку, домінує постачальник {dominant_supplier[0]}",
                'metric_value': metrics.market_concentration
            })
        
        # Аномалія: Цінова волатільність
        if metrics.price_volatility > 0.5:
            anomalies.append({
                'type': 'price_volatility',
                'severity': 'high' if metrics.price_volatility > 1.0 else 'medium',
                'description': "Надзвичайно висока волатільність цін",
                'metric_value': metrics.price_volatility
            })
        
        # Аномалія: Низька участь
        if metrics.avg_participants < 1.5:
            anomalies.append({
                'type': 'low_participation',
                'severity': 'medium',
                'description': "Низька середня кількість учасників у тендерах",
                'metric_value': metrics.avg_participants
            })
        
        # Аномалія: Нестабільність
        if metrics.market_stability < 0.2:
            anomalies.append({
                'type': 'market_instability',
                'severity': 'high',
                'description': "Високий рівень нестабільності ринку",
                'metric_value': metrics.market_stability
            })
        
        return anomalies