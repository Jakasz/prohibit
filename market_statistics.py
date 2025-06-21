# market_statistics.py
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np


class MarketStatistics:
    """
    Статистика ринку по категоріях та кластерах для оцінки нових постачальників
    """
    
    def __init__(self, category_manager=None):
        self.logger = logging.getLogger(__name__)
        self.category_manager = category_manager
        
        # Основні структури даних
        self.category_stats = {}
        self.cluster_stats = {}
        
        # Кеш для швидкого доступу
        self.percentile_cache = {}
        
        self.logger.info("✅ MarketStatistics ініціалізовано")
    
    def calculate_market_statistics_from_cache(self, cache_file: str = "all_data_cache.pkl") -> Dict[str, Any]:
        """
        Розрахунок статистики з кешу замість бази даних
        """
        import pickle
        
        self.logger.info(f"📂 Завантаження даних з {cache_file}...")
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Перевірка структури даних
            if isinstance(cached_data, dict) and 'data' in cached_data:
                historical_data = cached_data['data']
            elif isinstance(cached_data, list):
                historical_data = cached_data
            elif isinstance(cached_data, dict):
                # Структура з prf.py: {edrpou: [positions]}
                self.logger.info(f"✅ Завантажено дані для {len(cached_data)} постачальників")
                
                # Перетворюємо в плоский список позицій
                historical_data = []
                for edrpou, positions in cached_data.items():
                    for position in positions:
                        # Нормалізуємо структуру для обробки
                        normalized = {
                            'EDRPOU': position.get('edrpou', edrpou),
                            'F_TENDERNUMBER': position.get('tender_number', ''),
                            'F_INDUSTRYNAME': position.get('industry', ''),
                            'F_ITEMNAME': position.get('item_name', ''),
                            'WON': position.get('won', False),
                            'DATEEND': position.get('date_end', ''),
                            # Додаткові поля які можуть знадобитися
                            'supplier_name': position.get('supplier_name', ''),
                            'budget': position.get('budget', 0),
                            'quantity': position.get('quantity', 0),
                            'price': position.get('price', 0)
                        }
                        historical_data.append(normalized)
                
                self.logger.info(f"✅ Перетворено в {len(historical_data):,} записів")
            else:
                self.logger.error("Невідомий формат кешу")
                return {}
            
            # Використовуємо існуючий метод
            return self.calculate_market_statistics(historical_data)
            
        except Exception as e:
            self.logger.error(f"❌ Помилка завантаження кешу: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def calculate_market_statistics(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """
        Розрахунок повної статистики ринку
        """
        self.logger.info(f"📊 Розрахунок статистики для {len(historical_data):,} записів...")
        
        # Підготовка структур для збору даних
        category_data = defaultdict(lambda: {
            'tenders': defaultdict(set),  # tender_id -> set(suppliers)
            'suppliers': defaultdict(lambda: {'participated': 0, 'won': 0, 'first_seen': None}),
            'tender_dates': {},
            'positions_count': 0,
            'unique_items': set()
        })
        
        # Збір даних
        for item in historical_data:
            # ВАЖЛИВО: Використовуємо правильне поле для категорії
            # Пріоритет: primary_category -> category -> unknown
            category = item.get('primary_category')
            if not category or category == 'unknown':
                category = item.get('category')
            if not category or category == 'unknown':
                category = item.get('F_INDUSTRYNAME', item.get('category', 'unknown'))
            if not category:
                category = 'unknown'
            
            tender_id = item.get('F_TENDERNUMBER', item.get('tender_number', ''))
            edrpou = item.get('EDRPOU', item.get('edrpou', ''))
            
            if not tender_id or not edrpou:
                continue
            
            cat_data = category_data[category]
            
            # Збір даних по тендерах
            cat_data['tenders'][tender_id].add(edrpou)
            cat_data['positions_count'] += 1
            
            # Збір даних по постачальниках
            supplier_data = cat_data['suppliers'][edrpou]
            supplier_data['participated'] += 1
            
            if item.get('WON'):
                supplier_data['won'] += 1
            
            # Дата першої участі
            date_str = item.get('DATEEND')
            if date_str:
                try:
                    # Підтримка різних форматів дат
                    if '.' in date_str:
                        date = datetime.strptime(date_str, "%d.%m.%Y")
                    elif '-' in date_str:
                        date = datetime.strptime(date_str, "%Y-%m-%d")
                    else:
                        date = None
                        
                    if date:
                        if not supplier_data['first_seen'] or date < supplier_data['first_seen']:
                            supplier_data['first_seen'] = date
                        cat_data['tender_dates'][tender_id] = date
                except:
                    pass
            
            # Унікальні товари
            item_name = item.get('F_ITEMNAME', '')
            if item_name:
                cat_data['unique_items'].add(item_name.lower()[:50])  # Перші 50 символів
        
        # Розрахунок статистики по категоріях
        self.category_stats = {}
        
        for category, data in category_data.items():
            stats = self._calculate_category_metrics(category, data)
            self.category_stats[category] = stats
            
            self.logger.info(f"  {category}: {stats['total_tenders']} тендерів, "
                           f"{stats['total_suppliers']} постачальників")
        
        # Розрахунок статистики по кластерах
        if self.category_manager:
            self._calculate_cluster_statistics()
        
        # Збереження результатів
        self._save_statistics()
        
        return {
            'categories_processed': len(self.category_stats),
            'total_categories': len(category_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_category_metrics(self, category: str, data: Dict) -> Dict[str, Any]:
        """Розрахунок метрик для однієї категорії"""
        
        tenders = data['tenders']
        suppliers = data['suppliers']
        
        # Базові метрики
        total_tenders = len(tenders)
        total_suppliers = len(suppliers)
        total_positions = data['positions_count']
        
        if total_tenders == 0:
            return self._get_empty_stats()
        
        # 1. Конкурентне середовище
        suppliers_per_tender = [len(suppliers_set) for suppliers_set in tenders.values()]
        avg_suppliers = np.mean(suppliers_per_tender) if suppliers_per_tender else 0
        median_suppliers = np.median(suppliers_per_tender) if suppliers_per_tender else 0
        
        # 2. Розподіл перемог
        winners = [s for s, data in suppliers.items() if data['won'] > 0]
        unique_winners_ratio = len(winners) / total_suppliers if total_suppliers > 0 else 0
        
        # 3. Концентрація ринку
        market_shares = []
        for supplier, s_data in suppliers.items():
            share = s_data['participated'] / total_positions if total_positions > 0 else 0
            market_shares.append(share)
        
        hhi = sum(share**2 for share in market_shares) if market_shares else 0
        
        # 4. Топ гравці
        sorted_suppliers = sorted(suppliers.items(), 
                                key=lambda x: x[1]['won'], 
                                reverse=True)[:10]
        
        top3_wins = sum(s[1]['won'] for s in sorted_suppliers[:3])
        top3_participations = sum(s[1]['participated'] for s in sorted_suppliers[:3])
        top3_market_share = top3_participations / total_positions if total_positions > 0 else 0
        
        # 5. Статистика для новачків (перші 6 місяців)
        new_suppliers_data = self._analyze_new_suppliers(suppliers, data['tender_dates'])
        
        # 6. Win rate розподіл
        win_rate_distribution = self._calculate_win_rate_distribution(suppliers)
        
        # 7. Ймовірності
        empirical_win_probability = 1 / avg_suppliers if avg_suppliers > 0 else 0
        
        return {
            # Основні метрики
            'total_tenders': total_tenders,
            'total_suppliers': total_suppliers,
            'total_positions': total_positions,
            'unique_items_count': len(data['unique_items']),
            
            # Конкуренція
            'avg_suppliers_per_tender': round(avg_suppliers, 2),
            'median_suppliers_per_tender': median_suppliers,
            'competition_intensity': min(avg_suppliers / 10, 1.0) if avg_suppliers > 0 else 0,
            
            # Концентрація
            'market_concentration_hhi': round(hhi, 4),
            'unique_winners_ratio': round(unique_winners_ratio, 3),
            'top3_market_share': round(top3_market_share, 3),
            
            # Новачки
            'new_supplier_win_rate': new_suppliers_data['avg_win_rate'],
            'new_supplier_success_rate': new_suppliers_data['success_rate'],
            'new_suppliers_share': new_suppliers_data['market_share'],
            
            # Ймовірності
            'empirical_win_probability': round(empirical_win_probability, 3),
            'market_openness': round(1 - hhi, 3),  # Чим менше HHI, тим відкритіший ринок
            
            # Розподіли
            'win_rate_distribution': win_rate_distribution,
            'suppliers_count_by_experience': self._group_by_experience(suppliers),
            
            # Бар'єр входу (0-1, де 1 - високий)
            'entry_barrier_score': self._calculate_entry_barrier(
                avg_suppliers, hhi, new_suppliers_data['success_rate']
            )
        }
    
    def _analyze_new_suppliers(self, suppliers: Dict, tender_dates: Dict) -> Dict:
        """Аналіз результатів нових постачальників"""
        
        new_suppliers = {}
        cutoff_date = datetime.now() - timedelta(days=180)  # 6 місяців
        
        for edrpou, data in suppliers.items():
            if data['first_seen'] and data['first_seen'] > cutoff_date:
                new_suppliers[edrpou] = data
        
        if not new_suppliers:
            return {
                'count': 0,
                'avg_win_rate': 0.2,  # Дефолтне значення
                'success_rate': 0.15,
                'market_share': 0.0
            }
        
        # Розрахунки
        total_new = len(new_suppliers)
        winners = [s for s, d in new_suppliers.items() if d['won'] > 0]
        
        win_rates = []
        total_participations = 0
        
        for supplier, data in new_suppliers.items():
            if data['participated'] > 0:
                wr = data['won'] / data['participated']
                win_rates.append(wr)
                total_participations += data['participated']
        
        return {
            'count': total_new,
            'avg_win_rate': round(np.mean(win_rates), 3) if win_rates else 0.0,
            'success_rate': round(len(winners) / total_new, 3) if total_new > 0 else 0.0,
            'market_share': round(total_participations / sum(s['participated'] for s in suppliers.values()), 3)
        }
    
    def _calculate_win_rate_distribution(self, suppliers: Dict) -> Dict:
        """Розподіл постачальників за win rate"""
        
        distribution = {
            '0%': 0,
            '1-25%': 0,
            '26-50%': 0,
            '51-75%': 0,
            '76-100%': 0
        }
        
        for supplier, data in suppliers.items():
            if data['participated'] == 0:
                continue
                
            win_rate = data['won'] / data['participated']
            
            if win_rate == 0:
                distribution['0%'] += 1
            elif win_rate <= 0.25:
                distribution['1-25%'] += 1
            elif win_rate <= 0.50:
                distribution['26-50%'] += 1
            elif win_rate <= 0.75:
                distribution['51-75%'] += 1
            else:
                distribution['76-100%'] += 1
        
        return distribution
    
    def _group_by_experience(self, suppliers: Dict) -> Dict:
        """Групування постачальників за досвідом"""
        
        groups = {
            '1-5': {'count': 0, 'avg_win_rate': []},
            '6-20': {'count': 0, 'avg_win_rate': []},
            '21-50': {'count': 0, 'avg_win_rate': []},
            '51-100': {'count': 0, 'avg_win_rate': []},
            '100+': {'count': 0, 'avg_win_rate': []}
        }
        
        for supplier, data in suppliers.items():
            participations = data['participated']
            if participations == 0:
                continue
                
            win_rate = data['won'] / participations
            
            if participations <= 5:
                group = '1-5'
            elif participations <= 20:
                group = '6-20'
            elif participations <= 50:
                group = '21-50'
            elif participations <= 100:
                group = '51-100'
            else:
                group = '100+'
            
            groups[group]['count'] += 1
            groups[group]['avg_win_rate'].append(win_rate)
        
        # Розрахунок середніх
        for group, data in groups.items():
            if data['avg_win_rate']:
                data['avg_win_rate'] = round(np.mean(data['avg_win_rate']), 3)
            else:
                data['avg_win_rate'] = 0.0
        
        return groups
    
    def _calculate_entry_barrier(self, avg_suppliers: float, hhi: float, 
                                new_success_rate: float) -> float:
        """Розрахунок бар'єру входу (0-1)"""
        
        # Фактори бар'єру
        competition_factor = min(avg_suppliers / 10, 1.0) * 0.3
        concentration_factor = hhi * 0.4
        success_factor = (1 - new_success_rate) * 0.3
        
        barrier = competition_factor + concentration_factor + success_factor
        
        return round(min(barrier, 1.0), 3)
    
    def _calculate_cluster_statistics(self):
        """Розрахунок статистики по кластерах"""
        
        if not hasattr(self.category_manager, 'category_mappings'):
            return
        
        cluster_data = defaultdict(lambda: {
            'categories': set(),
            'suppliers': set(),
            'total_positions': 0,
            'total_wins': 0
        })
        
        # Збір даних по кластерах
        for category, stats in self.category_stats.items():
            # Знайти кластер для категорії
            for cluster_name, cluster_categories in self.category_manager.category_mappings.items():
                if category in cluster_categories:
                    cluster_data[cluster_name]['categories'].add(category)
                    cluster_data[cluster_name]['total_positions'] += stats['total_positions']
                    # Додаткова логіка...
        
        # Розрахунок метрик
        self.cluster_stats = {}
        for cluster, data in cluster_data.items():
            self.cluster_stats[cluster] = {
                'categories_count': len(data['categories']),
                'avg_competition': self._calculate_cluster_avg_competition(data['categories']),
                # Інші метрики...
            }
    
    def _calculate_cluster_avg_competition(self, categories: set) -> float:
        """Середня конкуренція в кластері"""
        competitions = []
        for cat in categories:
            if cat in self.category_stats:
                competitions.append(self.category_stats[cat]['avg_suppliers_per_tender'])
        
        return np.mean(competitions) if competitions else 0.0
    
    def get_category_context(self, category: str) -> Dict[str, Any]:
        """Отримання контексту категорії для прогнозування"""
        
        if category not in self.category_stats:
            # Повертаємо дефолтні значення
            return {
                'avg_suppliers_per_tender': 3.0,
                'empirical_win_probability': 0.33,
                'new_supplier_win_rate': 0.2,
                'market_openness': 0.7,
                'entry_barrier_score': 0.5
            }
        
        stats = self.category_stats[category]
        
        return {
            'avg_suppliers_per_tender': stats['avg_suppliers_per_tender'],
            'empirical_win_probability': stats['empirical_win_probability'],
            'new_supplier_win_rate': stats['new_supplier_win_rate'],
            'market_openness': stats['market_openness'],
            'entry_barrier_score': stats['entry_barrier_score'],
            'market_concentration': stats['market_concentration_hhi'],
            'top3_market_share': stats['top3_market_share']
        }
    
    def calculate_supplier_percentile(self, supplier_profile: Dict, category: str) -> float:
        """Розрахунок перцентиля постачальника в категорії"""
        
        if category not in self.category_stats:
            return 0.5  # Медіана за замовчуванням
        
        # Кеш-ключ
        cache_key = f"{supplier_profile.get('edrpou', '')}_{category}"
        if cache_key in self.percentile_cache:
            return self.percentile_cache[cache_key]
        
        # Розрахунок
        supplier_wr = supplier_profile.get('metrics', {}).get('win_rate', 0)
        distribution = self.category_stats[category]['win_rate_distribution']
        
        # Підрахунок постачальників з гіршим win rate
        worse_count = 0
        total_count = sum(distribution.values())
        
        if supplier_wr == 0:
            worse_count = 0
        elif supplier_wr <= 0.25:
            worse_count = distribution['0%']
        elif supplier_wr <= 0.50:
            worse_count = distribution['0%'] + distribution['1-25%']
        elif supplier_wr <= 0.75:
            worse_count = distribution['0%'] + distribution['1-25%'] + distribution['26-50%']
        else:
            worse_count = total_count - distribution['76-100%']
        
        percentile = worse_count / total_count if total_count > 0 else 0.5
        
        # Кешування
        self.percentile_cache[cache_key] = percentile
        
        return round(percentile, 3)
    
    def _get_empty_stats(self) -> Dict:
        """Порожня статистика для категорій без даних"""
        return {
            'total_tenders': 0,
            'total_suppliers': 0,
            'total_positions': 0,
            'avg_suppliers_per_tender': 0,
            'empirical_win_probability': 0,
            'new_supplier_win_rate': 0.2,
            'market_openness': 1.0,
            'entry_barrier_score': 0.0,
            'market_concentration_hhi': 0.0,
            'win_rate_distribution': {
                '0%': 0, '1-25%': 0, '26-50%': 0, '51-75%': 0, '76-100%': 0
            }
        }
    
    def _save_statistics(self, filepath: str = "market_statistics.json"):
        """Збереження статистики"""
        try:
            data = {
                'category_stats': self.category_stats,
                'cluster_stats': self.cluster_stats,
                'generated_at': datetime.now().isoformat(),
                'categories_count': len(self.category_stats)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Статистика збережена в {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Помилка збереження статистики: {e}")
    
    def load_statistics(self, filepath: str = "market_statistics.json") -> bool:
        """Завантаження збереженої статистики"""
        try:
            if not Path(filepath).exists():
                self.logger.warning(f"Файл {filepath} не знайдено")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.category_stats = data.get('category_stats', {})
            self.cluster_stats = data.get('cluster_stats', {})
            
            self.logger.info(f"✅ Завантажено статистику для {len(self.category_stats)} категорій")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Помилка завантаження статистики: {e}")
            return False