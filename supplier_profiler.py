import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass, field
import re


@dataclass
class SupplierMetrics:
    """Метрики постачальника"""
    total_tenders: int = 0
    won_tenders: int = 0
    total_positions: int = 0
    won_positions: int = 0
    total_revenue: float = 0.0
    avg_position_value: float = 0.0
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
    cpv_experience: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    brand_expertise: List[str] = field(default_factory=list)
    competitive_advantages: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    market_position: str = "unknown"
    reliability_score: float = 0.0
    profile_version: int = 1
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
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
                'total_revenue': self.metrics.total_revenue,
                'avg_position_value': self.metrics.avg_position_value,
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
            'cpv_experience': self.cpv_experience,
            'brand_expertise': self.brand_expertise,
            'competitive_advantages': self.competitive_advantages,
            'weaknesses': self.weaknesses,
            'market_position': self.market_position,
            'reliability_score': self.reliability_score,
            'profile_version': self.profile_version,
            'last_updated': self.last_updated
        }


class SupplierProfiler:
    """Система профілювання постачальників"""
    
    def __init__(self, category_manager=None):
        self.profiles: Dict[str, SupplierProfile] = {}
        self.category_manager = category_manager
        self.market_benchmarks: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
        
        # Патерни для розпізнавання
        self.brand_patterns = self._init_brand_patterns()
        self.quality_indicators = self._init_quality_indicators()
        
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
    
    def create_profile(self, supplier_data: List[Dict]) -> SupplierProfile:
        """Створення профілю постачальника"""
        if not supplier_data:
            return None
        
        # Базова інформація
        first_item = supplier_data[0]
        edrpou = first_item.get('EDRPOU', '')
        name = first_item.get('supp_name', '')
        
        profile = SupplierProfile(edrpou=edrpou, name=name)
        
        # Розрахунок метрик
        self._calculate_metrics(profile, supplier_data)
        self._analyze_categories(profile, supplier_data)
        self._analyze_industries(profile, supplier_data)
        self._analyze_cpv_experience(profile, supplier_data)
        self._analyze_brand_expertise(profile, supplier_data)
        self._determine_market_position(profile)
        self._identify_strengths_weaknesses(profile)
        self._calculate_reliability_score(profile, supplier_data)
        
        return profile
    
    def _calculate_metrics(self, profile: SupplierProfile, data: List[Dict]):
        """Розрахунок основних метрик"""
        metrics = profile.metrics
        
        # Підрахунок базових показників
        tender_numbers = set()
        won_tender_numbers = set()
        total_revenue = 0.0
        
        for item in data:
            tender_num = item.get('F_TENDERNUMBER')
            if tender_num:
                tender_numbers.add(tender_num)
                if item.get('WON'):
                    won_tender_numbers.add(tender_num)
            
            metrics.total_positions += 1
            if item.get('WON'):
                metrics.won_positions += 1
                try:
                    budget = float(item.get('ITEM_BUDGET', 0))
                    total_revenue += budget
                except:
                    pass
        
        metrics.total_tenders = len(tender_numbers)
        metrics.won_tenders = len(won_tender_numbers)
        metrics.total_revenue = total_revenue
        
        # Розрахунок win rates
        if metrics.total_tenders > 0:
            metrics.win_rate = metrics.won_tenders / metrics.total_tenders
        
        if metrics.total_positions > 0:
            metrics.position_win_rate = metrics.won_positions / metrics.total_positions
            metrics.avg_position_value = total_revenue / metrics.won_positions if metrics.won_positions > 0 else 0
        
        # Recent performance (останні 90 днів)
        self._calculate_recent_performance(profile, data)
        
        # Growth rate
        self._calculate_growth_rate(profile, data)
        
        # Stability score
        self._calculate_stability_score(profile, data)
    
    def _calculate_recent_performance(self, profile: SupplierProfile, data: List[Dict]):
        """Розрахунок недавньої продуктивності"""
        try:
            # Парсинг дат
            dated_items = []
            for item in data:
                date_str = item.get('DATEEND')
                if date_str:
                    try:
                        date = datetime.strptime(date_str, "%d.%m.%Y")
                        dated_items.append((date, item))
                    except:
                        pass
            
            if not dated_items:
                return
            
            dated_items.sort(key=lambda x: x[0], reverse=True)
            latest_date = dated_items[0][0]
            cutoff_date = latest_date - timedelta(days=90)
            
            # Фільтрація недавніх позицій
            recent_items = [item for date, item in dated_items if date >= cutoff_date]
            
            if recent_items:
                recent_won = sum(1 for item in recent_items if item.get('WON'))
                profile.metrics.recent_win_rate = recent_won / len(recent_items)
            
        except Exception as e:
            self.logger.error(f"Error calculating recent performance: {e}")
    
    def _calculate_growth_rate(self, profile: SupplierProfile, data: List[Dict]):
        """Розрахунок темпу зростання"""
        try:
            # Групування по кварталах
            quarterly_data = defaultdict(lambda: {'total': 0, 'won': 0})
            
            for item in data:
                date_str = item.get('DATEEND')
                if date_str:
                    try:
                        date = datetime.strptime(date_str, "%d.%m.%Y")
                        quarter_key = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
                        
                        quarterly_data[quarter_key]['total'] += 1
                        if item.get('WON'):
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
            
            for item in data:
                date_str = item.get('DATEEND')
                if date_str:
                    try:
                        date = datetime.strptime(date_str, "%d.%m.%Y")
                        quarter_key = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
                        
                        quarterly_data[quarter_key]['total'] += 1
                        if item.get('WON'):
                            quarterly_data[quarter_key]['won'] += 1
                    except:
                        pass
            
            for quarter_data in quarterly_data.values():
                if quarter_data['total'] > 0:
                    win_rate = quarter_data['won'] / quarter_data['total']
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
    
    def _analyze_categories(self, profile: SupplierProfile, data: List[Dict]):
        """Аналіз категорій постачальника"""
        category_stats = defaultdict(lambda: {
            'total': 0, 'won': 0, 'revenue': 0.0,
            'items': [], 'win_rate': 0.0
        })
        
        for item in data:
            item_name = item.get('F_ITEMNAME', '')
            
            # Визначення категорій
            if self.category_manager:
                categories = self.category_manager.categorize_item(item_name)
            else:
                categories = ['unknown']
            
            for category in categories:
                stats = category_stats[category]
                stats['total'] += 1
                stats['items'].append(item_name)
                
                if item.get('WON'):
                    stats['won'] += 1
                    try:
                        budget = float(item.get('ITEM_BUDGET', 0))
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
            
            if item.get('WON'):
                stats['won'] += 1
                try:
                    budget = float(item.get('ITEM_BUDGET', 0))
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
            cpv = item.get('CPV')
            if cpv and cpv != 0:
                cpv_str = str(cpv)
                stats = cpv_stats[cpv_str]
                stats['total'] += 1
                
                if item.get('WON'):
                    stats['won'] += 1
                
                # Додаємо категорії
                item_name = item.get('F_ITEMNAME', '')
                if self.category_manager:
                    categories = self.category_manager.categorize_item(item_name)
                    stats['categories'].update(categories)
        
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
            item_name = item.get('F_ITEMNAME', '')
            if item_name and item.get('WON'):
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
            if metrics.total_revenue >= 1000000:
                profile.market_position = "market_leader"
            else:
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
        
        # 4. Фінансовий об'єм
        if profile.metrics.total_revenue > 0:
            revenue_score = min(profile.metrics.total_revenue / 5000000, 1.0)
            factors.append(revenue_score * 0.2)
        
        profile.reliability_score = sum(factors)
    
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
        revenues = [p.metrics.total_revenue for p in category_profiles]
        positions = [p.metrics.total_positions for p in category_profiles]
        
        benchmarks = {
            'avg_win_rate': np.mean(win_rates),
            'median_win_rate': np.median(win_rates),
            'top_quartile_win_rate': np.percentile(win_rates, 75),
            'avg_revenue': np.mean(revenues),
            'median_revenue': np.median(revenues),
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