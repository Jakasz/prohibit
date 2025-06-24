import json
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass, field
import re

import tqdm


@dataclass
class SupplierMetrics:
    """Метрики постачальника"""
    total_tenders: int = 0
    won_tenders: int = 0
    total_positions: int = 0
    won_positions: int = 0
    win_rate: float = 0.0
    position_win_rate: float = 0.0
    recent_win_rate: float = 0.0
    growth_rate: float = 0.0
    stability_score: float = 0.0
    specialization_score: float = 0.0
    competition_resistance: float = 0.0


@dataclass
class SupplierProfile:
    """Повний профіль постачальника"""
    edrpou: str
    name: str
    metrics: SupplierMetrics = field(default_factory=SupplierMetrics)
    categories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    industries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    clusters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cpv_experience: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    brand_expertise: List[str] = field(default_factory=list)
    competitive_advantages: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    market_position: str = "unknown"
    reliability_score: float = 0.0
    profile_version: int = 1
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    risk_indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    has_risks: bool = False
    overall_risk_level: str = "low"
    top_competitors: List[Tuple[str, float]] = field(default_factory=list)
    bottom_competitors: List[Tuple[str, float]] = field(default_factory=list)    
    
    def to_dict(self) -> Dict:
        """Конвертація в словник"""
        return {
            'edrpou': self.edrpou,
            'name': self.name,
            'metrics': {
                'total_tenders': self.metrics.total_tenders,
                'won_tenders': self.metrics.won_tenders,
                'total_positions': self.metrics.total_positions,
                'won_positions': self.metrics.won_positions,
                'win_rate': self.metrics.win_rate,
                'position_win_rate': self.metrics.position_win_rate,
                'recent_win_rate': self.metrics.recent_win_rate,
                'growth_rate': self.metrics.growth_rate,
                'stability_score': self.metrics.stability_score,
                'specialization_score': self.metrics.specialization_score,
                'competition_resistance': self.metrics.competition_resistance
            },
            'categories': self.categories,
            'industries': self.industries,
            'clusters' : self.clusters,  
            'cpv_experience': self.cpv_experience,
            'brand_expertise': self.brand_expertise,
            'competitive_advantages': self.competitive_advantages,
            'weaknesses': self.weaknesses,
            'market_position': self.market_position,
            'reliability_score': self.reliability_score,
            'profile_version': self.profile_version,
            'last_updated': self.last_updated,
            'risk_indicators': self.risk_indicators,
            'has_risks': self.has_risks,
            'overall_risk_level': self.overall_risk_level,
            'top_competitors': self.top_competitors,
            'bottom_competitors': self.bottom_competitors
        }


class SupplierProfiler:
    """Система профілювання постачальників"""
    
    def __init__(self, categories_manager=None, vector_db=None):
        self.profiles: Dict[str, SupplierProfile] = {}
        self.categories_manager = categories_manager
        self.market_benchmarks: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
        
        # Патерни для розпізнавання
        self.brand_patterns = self._init_brand_patterns()
        self.quality_indicators = self._init_quality_indicators()
        self.vector_db = vector_db  # Додаємо посилання на векторну базу

    def build_profiles_from_aggregated_data(self, aggregated_data: Dict[str, List[Dict]], 
                                       batch_size: int = 1000) -> Dict[str, Any]:
        """
        Створення профілів з попередньо агрегованих даних
        Оптимізовано для швидкої обробки великих об'ємів
        """
        results = {
            'new_profiles': 0,
            'updated_profiles': 0,
            'errors': 0
        }
        
        # Обробка батчами для контролю пам'яті
        edrpou_list = list(aggregated_data.keys())
        
        for i in range(0, len(edrpou_list), batch_size):
            batch_edrpou = edrpou_list[i:i + batch_size]
            
            for edrpou in batch_edrpou:
                try:
                    items = aggregated_data[edrpou]
                    if not items:
                        continue
                    
                    # Швидке створення профілю
                    profile = self._create_minimal_profile(edrpou, items)
                    
                    if edrpou in self.profiles:
                        self.profiles[edrpou] = profile
                        results['updated_profiles'] += 1
                    else:
                        self.profiles[edrpou] = profile
                        results['new_profiles'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Помилка профілю {edrpou}: {e}")
                    results['errors'] += 1
            
            # Періодичне очищення пам'яті
            if i % (batch_size * 10) == 0:
                import gc
                gc.collect()
        
        return results

    def _create_minimal_profile(self, edrpou: str, items: List[Dict]) -> SupplierProfile:
        """Мінімальний профіль для швидкості"""
        profile = SupplierProfile(
            edrpou=edrpou,
            name=items[0].get('supplier_name', '') if items else ''
        )
        
        # Тільки базові метрики
        tenders = set()
        won = 0
        
        for item in items:
            tender = item.get('tender_number', '')
            if tender:
                tenders.add(tender)
            if item.get('won'):
                won += 1
        
        profile.metrics.total_positions = len(items)
        profile.metrics.won_positions = won
        profile.metrics.total_tenders = len(tenders)
        profile.metrics.position_win_rate = won / len(items) if items else 0
        
        return profile


    def _init_brand_patterns(self) -> Dict[str, re.Pattern]:
        """Ініціалізація патернів брендів"""
        brands = [
            "FENDT", "JOHN DEERE", "CASE", "NEW HOLLAND", "CLAAS",
            "CATERPILLAR", "KOMATSU", "VOLVO", "SCANIA", "MAN",
            "MERCEDES", "BMW", "BOSCH", "SIEMENS", "ABB"
        ]
        return {brand: re.compile(rf'\b{brand}\b', re.IGNORECASE) for brand in brands}
    
    def _init_quality_indicators(self) -> Dict[str, List[str]]:
        """Ініціалізація індикаторів якості"""
        return {
            'premium': ['оригінал', 'преміум', 'високоякіс', 'сертифікован'],
            'standard': ['стандарт', 'якісн', 'надійн'],
            'budget': ['економ', 'бюджет', 'аналог']
        }
    def build_profiles(self, update_mode: bool = False) -> Dict[str, Any]:
        """
        Масове створення/оновлення профілів постачальників з векторної бази tender_vectors
        Args:
            update_mode: True для оновлення існуючих профілів
        """
        self.logger.info(f"🔄 Завантаження даних з векторної бази tender_vectors...")

        # Отримання всіх унікальних EDRPOU з бази
        all_edrpou = self.vector_db.get_all_supplier_ids()  # Метод має повертати список EDRPOU

        self.logger.info(f"📊 Знайдено {len(all_edrpou)} унікальних постачальників у tender_vectors")

        results = {
            'total_suppliers': len(all_edrpou),
            'new_profiles': 0,
            'updated_profiles': 0,
            'errors': 0
        }

        for edrpou in tqdm(all_edrpou, desc="Створення профілів"):
            try:
                # Отримати агреговані дані по постачальнику з векторної бази
                supplier_data = self.vector_db.get_supplier_aggregate(edrpou)
                if not supplier_data:
                    continue

                if edrpou in self.profiles and update_mode:
                    self.update_profile_from_vector(edrpou, supplier_data)
                    results['updated_profiles'] += 1
                else:
                    profile = self.create_profile_from_vector(supplier_data)
                    if profile:
                        self.profiles[edrpou] = profile
                        results['new_profiles'] += 1
            except Exception as e:
                self.logger.error(f"Помилка створення профілю для {edrpou}: {e}")
                results['errors'] += 1

        self.logger.info(f"✅ Створено профілів: {results['new_profiles']}")
        self.logger.info(f"✅ Оновлено профілів: {results['updated_profiles']}")

        return results

    def create_profile_from_vector(self, supplier_data: Dict) -> SupplierProfile:
        """Створення профілю постачальника з агрегованих даних векторної бази"""
        edrpou = supplier_data.get('EDRPOU', '') or supplier_data.get('edrpou', '')
        name = supplier_data.get('supplier_name', '') or supplier_data.get('supp_name', '')
        profile = SupplierProfile(edrpou=edrpou, name=name)

        # Заповнення метрик з агрегованих даних
        metrics = profile.metrics
        metrics.total_tenders = supplier_data.get('total_tenders', 0)
        metrics.won_tenders = supplier_data.get('won_tenders', 0)
        metrics.total_positions = supplier_data.get('total_positions', 0)
        metrics.won_positions = supplier_data.get('won_positions', 0)
        metrics.win_rate = supplier_data.get('win_rate', 0.0)
        metrics.position_win_rate = supplier_data.get('position_win_rate', 0.0)
        metrics.recent_win_rate = supplier_data.get('recent_win_rate', 0.0)
        metrics.growth_rate = supplier_data.get('growth_rate', 0.0)
        metrics.stability_score = supplier_data.get('stability_score', 0.0)
        metrics.specialization_score = supplier_data.get('specialization_score', 0.0)
        metrics.competition_resistance = supplier_data.get('competition_resistance', 0.0)

        # Інші поля профілю
        profile.categories = supplier_data.get('categories', {})
        profile.industries = supplier_data.get('industries', {})
        profile.cpv_experience = supplier_data.get('cpv_experience', {})
        profile.brand_expertise = supplier_data.get('brand_expertise', [])
        profile.competitive_advantages = supplier_data.get('competitive_advantages', [])
        profile.weaknesses = supplier_data.get('weaknesses', [])
        profile.market_position = supplier_data.get('market_position', 'unknown')
        profile.reliability_score = supplier_data.get('reliability_score', 0.0)
        profile.profile_version = supplier_data.get('profile_version', 1)
        profile.last_updated = supplier_data.get('last_updated', datetime.now().isoformat())

        return profile

    def update_profile_from_vector(self, edrpou: str, supplier_data: Dict):
        """Оновлення профілю з агрегованих даних векторної бази"""
        profile = self.create_profile_from_vector(supplier_data)
        self.profiles[edrpou] = profile

    def get_all_profiles(self) -> Dict[str, SupplierProfile]:
        """Отримання всіх профілів"""
        return self.profiles

    def export_state(self) -> Dict[str, Any]:
        """Експорт стану профайлера"""
        return {
            'profiles': {
                edrpou: profile.to_dict() 
                for edrpou, profile in self.profiles.items()
            },
            'market_benchmarks': self.market_benchmarks
        }

    def load_state(self, state_data: Dict[str, Any]):
        """Завантаження стану профайлера"""
        # Очищення поточних профілів
        self.profiles.clear()
        
        # Завантаження профілів
        profiles_data = state_data.get('profiles', {})
        for edrpou, profile_data in profiles_data.items():
            # Відновлення профілю
            profile = SupplierProfile(
                edrpou=edrpou,
                name=profile_data.get('name', '')
            )
            
            # Відновлення метрик
            metrics_data = profile_data.get('metrics', {})
            for key, value in metrics_data.items():
                setattr(profile.metrics, key, value)
            
            # Відновлення інших полів
            profile.categories = profile_data.get('categories', {})
            profile.industries = profile_data.get('industries', {})
            profile.cpv_experience = profile_data.get('cpv_experience', {})
            profile.brand_expertise = profile_data.get('brand_expertise', [])
            profile.competitive_advantages = profile_data.get('competitive_advantages', [])
            profile.weaknesses = profile_data.get('weaknesses', [])
            profile.market_position = profile_data.get('market_position', 'unknown')
            profile.reliability_score = profile_data.get('reliability_score', 0.0)
            
            self.profiles[edrpou] = profile
        
        # Завантаження бенчмарків
        self.market_benchmarks = state_data.get('market_benchmarks', {})
        
        self.logger.info(f"✅ Завантажено {len(self.profiles)} профілів")

    
    def create_profile(self, edrpou, items):
        """Створення профілю постачальника"""
        if not items:
            return None
        
        # Діагностика даних
        self.logger.debug(f"Створення профілю для {edrpou} з {len(items)} записів")
        
        # Базова інформація - ПЕРЕВІРЯЄМО РІЗНІ ПОЛЯ
        first_item = items[0]
        
        # Витягуємо ЄДРПОУ з різних можливих полів
        actual_edrpou = (
            edrpou or  # Переданий параметр
            first_item.get('edrpou') or 
            first_item.get('EDRPOU') or
            ''
        )
        
        # Витягуємо назву з різних можливих полів
        name = (
            first_item.get('supplier_name') or 
            first_item.get('supp_name') or
            first_item.get('name') or
            first_item.get('SUPP_NAME') or
            f'Постачальник {actual_edrpou}'  # Fallback
        )
        
        self.logger.debug(f"Знайдено: ЄДРПОУ={actual_edrpou}, Назва={name}")
        
        profile = SupplierProfile(edrpou=actual_edrpou, name=name)
        
        # Розрахунок метрик
        self._calculate_metrics(profile, items)
        self._analyze_categories(profile, items)
        self._analyze_industries(profile, items)
        self._analyze_cpv_experience(profile, items)
        self._analyze_brand_expertise(profile, items)
        self._determine_market_position(profile)
        self._identify_strengths_weaknesses(profile)
        self._calculate_reliability_score(profile, items)
        self._analyze_risk_indicators(profile, items)
        
        return profile

    
    def _calculate_metrics(self, profile: SupplierProfile, data: List[Dict]):
        """Розрахунок основних метрик"""
        metrics = profile.metrics

        # Підрахунок базових показників
        tender_numbers = set()
        won_tender_numbers = set()

        for item in data:
            # ВИПРАВЛЕННЯ: Перевіряємо різні варіанти полів
            tender_num = (
                item.get('tender_number') or 
                item.get('F_TENDERNUMBER') or
                item.get('TENDER_NUMBER') or
                ''
            )
            
            if tender_num:
                tender_numbers.add(tender_num)
                
                # Перевіряємо різні варіанти поля WON
                is_won = (
                    item.get('won') or 
                    item.get('WON') or
                    False
                )
                
                if is_won:
                    won_tender_numbers.add(tender_num)

            metrics.total_positions += 1
            
            # Перевіряємо різні варіанти поля WON
            if item.get('won') or item.get('WON'):
                metrics.won_positions += 1

        metrics.total_tenders = len(tender_numbers)
        metrics.won_tenders = len(won_tender_numbers)

        # Діагностика
        self.logger.debug(f"Метрики: тендерів={metrics.total_tenders}, виграно={metrics.won_tenders}, "
                        f"позицій={metrics.total_positions}, виграно позицій={metrics.won_positions}")

        # Розрахунок win rates
        if metrics.total_tenders > 0:
            metrics.win_rate = metrics.won_tenders / metrics.total_tenders

        if metrics.total_positions > 0:
            metrics.position_win_rate = metrics.won_positions / metrics.total_positions

        # Recent performance (останні 180 днів)
        self._calculate_recent_performance(profile, data)

        # Growth rate
        self._calculate_growth_rate(profile, data)

        # Stability score
        self._calculate_stability_score(profile, data)

    
    def _calculate_recent_performance(self, profile: SupplierProfile, data: List[Dict]):
        """Розрахунок недавньої продуктивності (за 180 днів)"""
        try:
            # Парсинг дат
            dated_items = []
            for item in data:
                # ВИПРАВЛЕННЯ: Перевіряємо різні варіанти полів дати
                date_str = (
                    item.get('date_end') or
                    item.get('DATEEND') or
                    item.get('DATE_END') or
                    ''
                )
                
                if date_str:
                    try:
                        # Спробуємо різні формати дат
                        for fmt in ['%d.%m.%Y', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S']:
                            try:
                                date = datetime.strptime(date_str.split('T')[0], fmt)
                                dated_items.append((date, item))
                                break
                            except:
                                continue
                    except:
                        pass
        
            if not dated_items:
                return
            
            dated_items.sort(key=lambda x: x[0], reverse=True)
            latest_date = dated_items[0][0]
            cutoff_date = latest_date - timedelta(days=180)
            
            # Фільтрація недавніх позицій
            recent_items = [item for date, item in dated_items if date >= cutoff_date]
            
            if recent_items:
                recent_won = sum(1 for item in recent_items if (item.get('won') or item.get('WON')))
                profile.metrics.recent_win_rate = recent_won / len(recent_items)
            
        except Exception as e:
            self.logger.error(f"Error calculating recent performance: {e}")

    
    def _calculate_growth_rate(self, profile: SupplierProfile, data: List[Dict]):
        """Розрахунок темпу зростання"""
        try:
            # Групування по кварталах
            quarterly_data = defaultdict(lambda: {'total': 0, 'won': 0})
            
            for item in data:
                date_str = item.get('DATEEND') or item.get('date_end')
                if date_str:
                    try:
                        date = datetime.strptime(date_str, "%d.%m.%Y")
                        quarter_key = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
                        
                        quarterly_data[quarter_key]['total'] += 1
                        if (item.get('WON') or item.get('won')):
                            quarterly_data[quarter_key]['won'] += 1
                    except:
                        pass
            
            if len(quarterly_data) >= 2:
                quarters = sorted(quarterly_data.keys())
                
                # Порівняння останнього та передостаннього кварталів
                last_quarter = quarterly_data[quarters[-1]]
                prev_quarter = quarterly_data[quarters[-2]]
                
                if prev_quarter['total'] > 0:
                    growth = (last_quarter['total'] - prev_quarter['total']) / prev_quarter['total']
                    profile.metrics.growth_rate = growth
            
        except Exception as e:
            self.logger.error(f"Error calculating growth rate: {e}")
    
    def _calculate_stability_score(self, profile: SupplierProfile, data: List[Dict]):
        """Розрахунок стабільності постачальника"""
        try:
            # Фактори стабільності
            factors = []
            
            # 1. Стабільність win rate по кварталах
            quarterly_win_rates = []
            quarterly_data = defaultdict(lambda: {'total': 0, 'won': 0})
            42934830
            for item in data:
                date_str = (item.get('DATEEND') or 
                            item.get('date_end')) 
                if date_str:
                    try:
                        date = datetime.strptime(date_str, "%d.%m.%Y")
                        quarter_key = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
                        
                        quarterly_data[quarter_key]['total'] += 1
                        if item.get('WON') or item.get('won'):
                            quarterly_data[quarter_key]['won'] += 1
                    except:
                        pass
            
            for quarter_data in quarterly_data.values():
                if quarter_data['total'] > 0:
                    win_rate = quarter_data['won']  / quarter_data['total']
                    quarterly_win_rates.append(win_rate)
            
            if len(quarterly_win_rates) >= 2:
                # Низька варіація = висока стабільність
                variance = np.var(quarterly_win_rates)
                stability_from_variance = 1.0 - min(variance * 10, 1.0)
                factors.append(stability_from_variance)
            
            # 2. Регулярність участі
            if len(quarterly_data) >= 4:
                participation_regularity = len(quarterly_data) / 4  # Нормалізовано до року
                factors.append(min(participation_regularity, 1.0))
            
            # 3. Досвід (кількість тендерів)
            experience_factor = min(profile.metrics.total_tenders / 50, 1.0)
            factors.append(experience_factor)
            
            # Загальний показник стабільності
            if factors:
                profile.metrics.stability_score = np.mean(factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating stability score: {e}")
    
    def _analyze_categories(self, profile, items):
        """Аналіз категорій постачальника"""
        category_stats = defaultdict(lambda: {
            'total': 0, 'won': 0, 'revenue': 0.0,
            'items': [], 'win_rate': 0.0
        })
        
        for item in items:
            # ВИПРАВЛЕННЯ: Перетворюємо рядок на список
            category = (item.get('industry') or item.get('F_INDUSTRYNAME'))
            nomenclature = item.get('item_name', '') 
            if not category:
                categories = ['unknown']
            else:
                # Якщо це рядок - робимо список з одним елементом
                categories = [category]

            for category in categories:
                stats = category_stats[category]
                stats['total'] += 1
                stats['items'].append(nomenclature)  # Додаємо назву позиції

                if (item.get('won') or item.get('WON')):
                    stats['won'] += 1
                    try:
                        budget = float(item.get('budget') or item.get('ITEM_BUDGET'))*float(item.get('currency_rate') or item.get('F_TENDERCURRENCYRATE'))
                        stats['revenue'] += budget
                    except:
                        pass
        
        # Розрахунок win rate та спеціалізації
        total_positions = sum(stats['total'] for stats in category_stats.values())
        
        for category, stats in category_stats.items():
            if stats['total'] > 0:
                stats['win_rate'] = stats['won'] / stats['total']
                stats['specialization'] = stats['total'] / total_positions
                
                # Зберігаємо тільки топ-5 товарів для економії пам'яті
                stats['top_items'] = list(set(stats['items']))[:5]
                del stats['items']
        
        profile.categories = dict(category_stats)
        
        # Розрахунок спеціалізації (HHI для категорій)
        if profile.categories:
            shares = [stats['specialization'] for stats in profile.categories.values()]
            profile.metrics.specialization_score = sum(s**2 for s in shares)

    
    def _analyze_industries(self, profile: SupplierProfile, data: List[Dict]):
        """Аналіз індустрій постачальника"""
        industry_stats = defaultdict(lambda: {
            'total': 0, 'won': 0, 'revenue': 0.0, 'win_rate': 0.0
        })
        
        for item in data:
            industry = item.get('F_INDUSTRYNAME', 'unknown')
            stats = industry_stats[industry]
            stats['total'] += 1
            
            if item.get('won'):
                stats['won'] += 1
                try:
                    budget = float(item.get('budget', 0))
                    stats['revenue'] += budget
                except:
                    pass
        
        # Розрахунок win rate
        for industry, stats in industry_stats.items():
            if stats['total'] > 0:
                stats['win_rate'] = stats['won'] / stats['total']
        
        profile.industries = dict(industry_stats)
    
    def _analyze_cpv_experience(self, profile: SupplierProfile, data: List[Dict]):
        """Аналіз досвіду з CPV кодами"""
        cpv_stats = defaultdict(lambda: {
            'total': 0, 'won': 0, 'win_rate': 0.0,
            'categories': set()
        })
        
        for item in data:
            cpv = item.get('CPV', '')
            if not cpv:
                cpv = item.get('cpv')
            if cpv and cpv != 0:
                cpv_str = str(cpv)
                stats = cpv_stats[cpv_str]
                stats['total'] += 1

                if (item.get('won') or item.get('WON')): 
                    stats['won'] += 1
                
            #     # Додаємо категорії
            #     item_name = (item.get('item_name',) or 
            #                  item.get('F_ITEMNAME', ''))
            #     if self.categories_manager:
            #         categories = self.categories_manager.categorize_item(item_name)
            #         stats['categories'].update(categories)
        
        # Розрахунок win rate та конвертація sets в lists
        for cpv, stats in cpv_stats.items():
            if stats['total'] > 0:
                stats['win_rate'] = stats['won'] / stats['total']
            stats['categories'] = list(stats['categories'])
        
        profile.cpv_experience = dict(cpv_stats)
    
    def _analyze_brand_expertise(self, profile: SupplierProfile, data: List[Dict]):
        """Аналіз досвіду з брендами"""
        brand_counter = Counter()
        
        for item in data:
            item_name = item.get('F_ITEMNAME', '') or item.get('item_name', '')
            if item_name and (item.get('WON') or item.get('won')):
                for brand, pattern in self.brand_patterns.items():
                    if pattern.search(item_name):
                        brand_counter[brand] += 1
        
        # Топ бренди
        profile.brand_expertise = [brand for brand, _ in brand_counter.most_common(10)]
    
    def _determine_market_position(self, profile: SupplierProfile):
        """Визначення позиції на ринку"""
        metrics = profile.metrics
        
        # Критерії для визначення позиції
        if metrics.total_tenders >= 50 and metrics.win_rate >= 0.3:
            profile.market_position = "established_player"
        elif metrics.total_tenders >= 20:
            if metrics.win_rate >= 0.2:
                profile.market_position = "competitive_player"
            else:
                profile.market_position = "active_participant"
        elif metrics.total_tenders >= 5:
            profile.market_position = "emerging_player"
        else:
            profile.market_position = "new_entrant"
    
    def _identify_strengths_weaknesses(self, profile: SupplierProfile):
        """Ідентифікація сильних та слабких сторін"""
        strengths = []
        weaknesses = []
        metrics = profile.metrics
        
        # Сильні сторони
        if metrics.win_rate >= 0.3:
            strengths.append("Високий відсоток перемог")
        
        if metrics.stability_score >= 0.7:
            strengths.append("Стабільна діяльність")
        
        if metrics.specialization_score >= 0.5:
            strengths.append("Чітка спеціалізація")
        
        if profile.brand_expertise:
            strengths.append(f"Досвід з брендами: {', '.join(profile.brand_expertise[:3])}")
        
        if metrics.recent_win_rate > metrics.win_rate * 1.2:
            strengths.append("Покращення результатів")
        
        # Слабкі сторони
        if metrics.win_rate < 0.1:
            weaknesses.append("Низький відсоток перемог")
        
        if metrics.growth_rate < -0.2:
            weaknesses.append("Зниження активності")
        
        if metrics.specialization_score < 0.2:
            weaknesses.append("Відсутність чіткої спеціалізації")
        
        if metrics.recent_win_rate < metrics.win_rate * 0.8:
            weaknesses.append("Погіршення результатів")
        
        profile.competitive_advantages = strengths
        profile.weaknesses = weaknesses
    
    def _calculate_reliability_score(self, profile: SupplierProfile, data: List[Dict]):
        """Розрахунок показника надійності"""
        factors = []

        # 1. Досвід (кількість тендерів)
        experience_score = min(profile.metrics.total_tenders / 100, 1.0)
        factors.append(experience_score * 0.3)

        # 2. Win rate
        win_rate_score = min(profile.metrics.win_rate * 2, 1.0)
        factors.append(win_rate_score * 0.3)

        # 3. Стабільність
        factors.append(profile.metrics.stability_score * 0.2)

        profile.reliability_score = sum(factors)

    def _analyze_risk_indicators(self, profile: SupplierProfile, data: List[Dict]):
        """Аналіз ризик-індикаторів для профілю"""
        risk_indicators = {}
        
        # 1. Ранній попереджувальний індикатор (падіння win rate)
        early_warning = self._check_early_warning_indicator(profile, data)
        if early_warning:
            risk_indicators['early_warning'] = early_warning
        
        # 2. Концентраційний ризик (залежність від одного замовника)
        concentration_risk = self._check_concentration_risk(data)
        if concentration_risk:
            risk_indicators['concentration_risk'] = concentration_risk
        
        # 3. Ризик зникнення
        disappearance_risk = self._check_disappearance_risk(data)
        if disappearance_risk:
            risk_indicators['disappearance_risk'] = disappearance_risk
        
        # Додаємо до профілю
        if risk_indicators:
            profile.risk_indicators = risk_indicators
            profile.has_risks = True
            # Визначаємо загальний рівень ризику
            risk_levels = [r.get('level', 'low') for r in risk_indicators.values()]
            if 'critical' in risk_levels:
                profile.overall_risk_level = 'critical'
            elif 'high' in risk_levels:
                profile.overall_risk_level = 'high'
            elif 'medium' in risk_levels:
                profile.overall_risk_level = 'medium'
            else:
                profile.overall_risk_level = 'low'
        else:
            profile.has_risks = False
            profile.overall_risk_level = 'low'

    def _check_early_warning_indicator(self, profile: SupplierProfile, data: List[Dict]):
        """Перевірка різкого падіння win rate"""
        # Групуємо дані по кварталах
        quarterly_data = defaultdict(lambda: {'total': 0, 'won': 0})
        
        for item in data:
            date_str = item.get('DATEEND') or item.get('date_end')
            if date_str:
                try:
                    date = datetime.strptime(date_str, "%d.%m.%Y")
                    quarter_key = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
                    quarterly_data[quarter_key]['total'] += 1
                    if item.get('WON') or item.get('won'):
                        quarterly_data[quarter_key]['won'] += 1
                except:
                    pass
        
        if len(quarterly_data) < 2:
            return None
        
        # Сортуємо квартали
        sorted_quarters = sorted(quarterly_data.keys())
        
        # Порівнюємо останні 2 квартали
        current_q = sorted_quarters[-1]
        previous_q = sorted_quarters[-2]
        
        current_data = quarterly_data[current_q]
        previous_data = quarterly_data[previous_q]
        
        if current_data['total'] > 0 and previous_data['total'] > 0:
            current_wr = current_data['won'] / current_data['total']
            previous_wr = previous_data['won'] / previous_data['total']
            
            if previous_wr > 0:
                decline = (previous_wr - current_wr) / previous_wr
                
                if decline > 0.3:  # Падіння більше 30%
                    return {
                        'level': 'high',
                        'message': f'Різке падіння win rate з {previous_wr:.1%} до {current_wr:.1%} (-{decline:.1%})',
                        'previous_quarter': previous_q,
                        'current_quarter': current_q,
                        'previous_win_rate': previous_wr,
                        'current_win_rate': current_wr,
                        'decline_percentage': decline
                    }
        
        return None  

    def _check_concentration_risk(self, data: List[Dict]):
        """Перевірка залежності від одного замовника"""
        owner_counts = defaultdict(int)
        total_positions = len(data)
        
        for item in data:
            owner = item.get('OWNER_NAME') or item.get('owner_name')
            if owner:
                owner_counts[owner] += 1
        
        if not owner_counts or total_positions == 0:
            return None
        
        # Знаходимо топового замовника
        top_owner, top_count = max(owner_counts.items(), key=lambda x: x[1])
        concentration = top_count / total_positions
        
        if concentration >= 0.75:  # 75% або більше
            return {
                'level': 'high',
                'message': f'Критична залежність від замовника "{top_owner}": {concentration:.1%} всіх тендерів',
                'dominant_owner': top_owner,
                'concentration_percentage': concentration,
                'positions_count': top_count,
                'total_positions': total_positions,
                'other_owners': len(owner_counts) - 1
            }
        
        return None
      
    def _check_disappearance_risk(self, data: List[Dict]):
        """Перевірка ризику зникнення (різке зниження активності)"""
        # Групуємо по роках
        yearly_activity = defaultdict(int)
        latest_date = None
        
        for item in data:
            date_str = item.get('DATEEND') or item.get('date_end')
            if date_str:
                try:
                    date = datetime.strptime(date_str, "%d.%m.%Y")
                    year = date.year
                    yearly_activity[year] += 1
                    
                    if latest_date is None or date > latest_date:
                        latest_date = date
                except:
                    pass
        
        if len(yearly_activity) < 2:
            return None
        
        # Сортуємо роки
        sorted_years = sorted(yearly_activity.keys())
        
        # Порівнюємо останні 2 роки
        if len(sorted_years) >= 2:
            current_year = sorted_years[-1]
            previous_year = sorted_years[-2]
            
            current_activity = yearly_activity[current_year]
            previous_activity = yearly_activity[previous_year]
            
            if previous_activity > 0:
                decline = (previous_activity - current_activity) / previous_activity
                
                if decline > 0.7:  # Падіння більше 70%
                    # Перевіряємо чи давно остання активність
                    days_since_last = (datetime.now() - latest_date).days if latest_date else 999
                    
                    if days_since_last > 180:  # Більше 6 місяців
                        return {
                            'level': 'critical',
                            'message': f'Постачальник зникає! Активність впала з {previous_activity} до {current_activity} позицій/рік. Остання активність {days_since_last} днів тому',
                            'previous_year': previous_year,
                            'current_year': current_year,
                            'previous_activity': previous_activity,
                            'current_activity': current_activity,
                            'last_activity_date': latest_date.strftime('%d.%m.%Y') if latest_date else 'Невідомо',
                            'days_inactive': days_since_last
                        }
                    else:
                        return {
                            'level': 'medium',
                            'message': f'Різке зниження активності з {previous_activity} до {current_activity} позицій/рік (-{decline:.1%})',
                            'previous_year': previous_year,
                            'current_year': current_year,
                            'decline_percentage': decline
                        }
        
        return None


    
    def update_profile(self, edrpou: str, new_data: List[Dict]):
        """Оновлення профілю новими даними"""
        if edrpou in self.profiles:
            # Об'єднуємо з існуючими даними
            existing_profile = self.profiles[edrpou]
            existing_profile.profile_version += 1
            existing_profile.last_updated = datetime.now().isoformat()
            
            # Перераховуємо профіль з усіма даними
            # (в реальній системі тут був би більш ефективний інкрементальний підхід)
            updated_profile = self.create_profile(new_data)
            self.profiles[edrpou] = updated_profile
        else:
            # Створюємо новий профіль
            profile = self.create_profile(new_data)
            if profile:
                self.profiles[edrpou] = profile
    
    def get_profile(self, edrpou: str) -> Optional[SupplierProfile]:
        """Отримання профілю постачальника"""
        return self.profiles.get(edrpou)
    
    def get_similar_suppliers(self, edrpou: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Пошук схожих постачальників"""
        target_profile = self.profiles.get(edrpou)
        if not target_profile:
            return []
        
        similarities = []
        
        for other_edrpou, other_profile in self.profiles.items():
            if other_edrpou == edrpou:
                continue
            
            # Розрахунок схожості
            similarity = self._calculate_profile_similarity(target_profile, other_profile)
            similarities.append((other_edrpou, similarity))
        
        # Сортування за схожістю
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _calculate_profile_similarity(self, profile1: SupplierProfile, profile2: SupplierProfile) -> float:
        """Розрахунок схожості двох профілів"""
        similarity_scores = []
        
        # 1. Схожість категорій
        cats1 = set(profile1.categories.keys())
        cats2 = set(profile2.categories.keys())
        if cats1 or cats2:
            category_similarity = len(cats1.intersection(cats2)) / len(cats1.union(cats2))
            similarity_scores.append(category_similarity)
        
        # 2. Схожість індустрій
        ind1 = set(profile1.industries.keys())
        ind2 = set(profile2.industries.keys())
        if ind1 or ind2:
            industry_similarity = len(ind1.intersection(ind2)) / len(ind1.union(ind2))
            similarity_scores.append(industry_similarity)
        
        # 3. Схожість метрик
        metric_diff = abs(profile1.metrics.win_rate - profile2.metrics.win_rate)
        metric_similarity = 1.0 - min(metric_diff * 2, 1.0)
        similarity_scores.append(metric_similarity)
        
        # 4. Схожість позиції на ринку
        if profile1.market_position == profile2.market_position:
            similarity_scores.append(1.0)
        else:
            similarity_scores.append(0.5)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def get_market_benchmarks(self, category: str = None) -> Dict[str, float]:
        """Отримання ринкових бенчмарків"""
        if category:
            # Бенчмарки для конкретної категорії
            category_profiles = [
                p for p in self.profiles.values()
                if category in p.categories
            ]
        else:
            # Загальні бенчмарки
            category_profiles = list(self.profiles.values())

        if not category_profiles:
            return {}

        # Розрахунок бенчмарків
        win_rates = [p.metrics.win_rate for p in category_profiles]
        positions = [p.metrics.total_positions for p in category_profiles]

        benchmarks = {
            'avg_win_rate': np.mean(win_rates),
            'median_win_rate': np.median(win_rates),
            'top_quartile_win_rate': np.percentile(win_rates, 75),
            'avg_positions': np.mean(positions),
            'total_suppliers': len(category_profiles)
        }

        return benchmarks
    
    def save_profiles(self, filepath: str):
        """Збереження профілів"""
        profiles_data = {
            edrpou: profile.to_dict()
            for edrpou, profile in self.profiles.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Збережено {len(profiles_data)} профілів постачальників")
    
    def load_profiles(self, filepath: str):
        """Завантаження профілів"""
        try:
            # Перевірка різних версій файлу
            if not Path(filepath).exists():
                # Спробуємо файл з кластерами
                clusters_file = filepath.replace('.json', '_with_clusters.json')
                if Path(clusters_file).exists():
                    filepath = clusters_file
                    self.logger.info(f"Використовуємо файл з кластерами: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                profiles_data = json.load(f)
            
            self.profiles = {}
            for edrpou, data in profiles_data.items():
                # Відновлення профілю з даних
                profile = SupplierProfile(
                    edrpou=edrpou,
                    name=data.get('name', '')
                )
                
                # Відновлення метрик
                metrics_data = data.get('metrics', {})
                for key, value in metrics_data.items():
                    setattr(profile.metrics, key, value)
                
                # Відновлення інших полів
                profile.categories = data.get('categories', {})
                profile.industries = data.get('industries', {})
                profile.cpv_experience = data.get('cpv_experience', {})
                profile.brand_expertise = data.get('brand_expertise', [])
                profile.competitive_advantages = data.get('competitive_advantages', [])
                profile.weaknesses = data.get('weaknesses', [])
                profile.market_position = data.get('market_position', 'unknown')
                profile.reliability_score = data.get('reliability_score', 0.0)
                profile.profile_version = data.get('profile_version', 1)
                profile.last_updated = data.get('last_updated', '')
                
                self.profiles[edrpou] = profile
            
            self.logger.info(f"Завантажено {len(self.profiles)} профілів постачальників")
            
        except Exception as e:
            self.logger.error(f"Помилка завантаження профілів: {e}")