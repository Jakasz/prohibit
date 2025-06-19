import json
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np


class CategoryManager:
    """
    Менеджер категорій тендерів з аналізом конкуренції
    
    Функції:
    - Завантаження та управління категоріями з JSONL
    - Автоматична категоризація товарів/послуг
    - Аналіз конкуренції в категоріях
    - Динамічне оновлення категорій
    """
    
    def __init__(self, categories_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Основні дані
        self.categories = {}  # {category_id: category_data}
        self.category_keywords = {}  # {category_id: [keywords]}
        self.category_patterns = {}  # {category_id: compiled_regex}
        
        # Статистика конкуренції по категоріях
        self.category_competition = defaultdict(lambda: {
            'total_tenders': 0,
            'total_suppliers': 0,
            'avg_suppliers_per_tender': 0,
            'competition_intensity': 0,  # 0-1 (низька-висока)
            'avg_win_rate': 0,
            'price_volatility': 0,
            'entry_barrier': 0,  # 0-1 (низький-високий)
            'market_concentration': 0,  # 0-1 (розпорошений-монополізований)
            'top_suppliers': [],
            'recent_trends': 'stable'
        })
        
        # Дані для аналізу
        self.tender_history = []  # Історія для аналізу трендів
        self.supplier_category_performance = defaultdict(lambda: defaultdict(dict))
        
        # Базові категорії (fallback)
        self.base_categories = self._init_base_categories()
        
        # Завантаження категорій з файлу
        if categories_file:
            self.load_categories_from_file(categories_file)
        
        self.logger.info("✅ CategoryManager ініціалізовано")
    
    def _init_base_categories(self) -> Dict[str, Dict]:
        """Ініціалізація базових категорій"""
        return {
            "agricultural_parts": {
                "id": "agricultural_parts",
                "name": "Сільськогосподарські запчастини",
                "active": True,
                "keywords": ["запчаст", "втулка", "підшипник", "фільтр", "ремінь", 
                           "поршень", "клапан", "насос", "гідравл", "трактор", "комбайн"],
                "regex_patterns": [r"\b(запчаст|втулк|підшипник)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "medium"
            },
            "electronics": {
                "id": "electronics",
                "name": "Електроніка та електротехніка",
                "active": True,
                "keywords": ["кабель", "роз'єм", "електр", "провід", "трансформатор", 
                           "конденсатор", "резистор", "діод", "світлодіод", "лампа"],
                "regex_patterns": [r"\b(електр|кабель|провід)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "high"
            },
            "construction": {
                "id": "construction",
                "name": "Будівельні матеріали",
                "active": True,
                "keywords": ["цемент", "бетон", "арматура", "цегла", "плитка", 
                           "фарба", "ізоляція", "покрівл", "сталь", "дерев"],
                "regex_patterns": [r"\b(цемент|бетон|арматур)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "medium"
            },
            "office_supplies": {
                "id": "office_supplies",
                "name": "Канцелярські товари та офісне обладнання",
                "active": True,
                "keywords": ["папір", "ручка", "картридж", "тонер", "канцел", 
                           "меблі", "стіл", "стілець", "принтер", "сканер"],
                "regex_patterns": [r"\b(папір|картридж|канцел)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "high"
            },
            "medical": {
                "id": "medical",
                "name": "Медичні товари та обладнання",
                "active": True,
                "keywords": ["медичн", "шприц", "бинт", "маска", "рукавич", 
                           "дезінфект", "ліки", "препарат", "апарат", "діагност"],
                "regex_patterns": [r"\b(медичн|шприц|препарат)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "low"
            },
            "food": {
                "id": "food",
                "name": "Продукти харчування",
                "active": True,
                "keywords": ["продукт", "молоко", "хліб", "м'ясо", "овоч", "фрукт", 
                           "крупа", "цукор", "олія", "консерв", "харчов"],
                "regex_patterns": [r"\b(продукт|молоко|харчов)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "medium"
            },
            "fuel": {
                "id": "fuel",
                "name": "Паливо та мастильні матеріали",
                "active": True,
                "keywords": ["паливо", "бензин", "дизель", "газ", "мазут", "солярка", 
                           "масло", "мастил", "антифриз"],
                "regex_patterns": [r"\b(паливо|бензин|дизель)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "low"
            },
            "services": {
                "id": "services",
                "name": "Послуги",
                "active": True,
                "keywords": ["послуг", "ремонт", "обслуговуван", "консультац", 
                           "проектуван", "будівництв", "транспорт", "логіст"],
                "regex_patterns": [r"\b(послуг|ремонт|консультац)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "high"
            },
            "communication": {
                "id": "communication", 
                "name": "Расходы на связь, канали даних, інтернет",
                "active": True,
                "keywords": ["зв'язок", "інтернет", "телефон", "мобільн", "канал", 
                           "передач", "даних", "хостинг", "домен", "wifi", "ethernet"],
                "regex_patterns": [r"\b(зв'язок|інтернет|телефон)\w*\b"],
                "parent_category": None,
                "subcategories": [],
                "competition_level": "medium"
            }
        }
    

    def load_category_mappings(self, mapping_file: str = "data/category_map.json"):
        """Завантаження маппінгу категорій"""
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.category_mappings = json.load(f)
            
            # Створюємо зворотній mapping
            self.reverse_mapping = {}
            for canonical, variants in self.category_mappings.items():
                for variant in variants:
                    self.reverse_mapping[variant.lower().strip()] = canonical
            
            self.logger.info(f"✅ Завантажено {len(self.category_mappings)} канонічних категорій")
            return True
        except Exception as e:
            self.logger.error(f"❌ Помилка завантаження маппінгу: {e}")
            return False

    def get_canonical_category(self, category_name: str) -> str:
        """Отримання канонічної назви категорії"""
        if not hasattr(self, 'reverse_mapping'):
            return category_name
        
        category_lower = category_name.lower().strip()
        return self.reverse_mapping.get(category_lower, category_name)

    def analyze_owner_patterns(self, historical_data: List[Dict]) -> Dict[str, Dict]:
        """Аналіз патернів по власниках"""
        owner_stats = defaultdict(lambda: {
            'categories': defaultdict(int),
            'industries': defaultdict(int),
            'items_count': 0,
            'unique_suppliers': set(),
            'total_budget': 0.0
        })
        
        for item in historical_data:
            owner = item.get('OWNER_NAME', 'unknown')
            stats = owner_stats[owner]
            
            # Категоризація
            item_name = item.get('F_ITEMNAME', '')
            if item_name:
                categories = self.categorize_item(item_name)
                if categories:
                    primary_cat = categories[0][0]
                    canonical_cat = self.get_canonical_category(primary_cat)
                    stats['categories'][canonical_cat] += 1
            
            # Інші метрики
            stats['industries'][item.get('F_INDUSTRYNAME', 'unknown')] += 1
            stats['items_count'] += 1
            stats['unique_suppliers'].add(item.get('EDRPOU', ''))
            
            try:
                budget = float(item.get('ITEM_BUDGET', 0))
                stats['total_budget'] += budget
            except:
                pass
    
        # Конвертація для серіалізації
        result = {}
        for owner, stats in owner_stats.items():
            result[owner] = {
                'categories': dict(stats['categories']),
                'industries': dict(stats['industries']),
                'items_count': stats['items_count'],
                'unique_suppliers': len(stats['unique_suppliers']),
                'total_budget': stats['total_budget']
            }
        
        return result


    
    def load_categories_from_file(self, filepath: str) -> bool:
        """
        Завантаження категорій з JSONL файлу
        
        Формат файлу:
        {"category": "Расходы на связь, канали данных, интернет", "active": true}
        {"category": "Електроніка", "active": false}
        """
        try:
            if not Path(filepath).exists():
                self.logger.warning(f"Файл категорій {filepath} не знайдено. Використовуються базові категорії")
                self.categories = self.base_categories.copy()
                self._compile_patterns()
                return False
            
            loaded_categories = {}
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        category_data = json.loads(line)
                        
                        # Генерація ID з назви
                        category_name = category_data.get('category', '').strip()
                        if not category_name:
                            continue
                        
                        category_id = self._generate_category_id(category_name)
                        
                        # Створення повної структури категорії
                        full_category = {
                            "id": category_id,
                            "name": category_name,
                            "active": category_data.get('active', True),
                            "keywords": self._extract_keywords_from_name(category_name),
                            "regex_patterns": self._generate_regex_patterns(category_name),
                            "parent_category": category_data.get('parent_category'),
                            "subcategories": category_data.get('subcategories', []),
                            "competition_level": category_data.get('competition_level', 'medium'),
                            "source": "file",
                            "created_at": datetime.now().isoformat()
                        }
                        
                        loaded_categories[category_id] = full_category
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Помилка парсингу JSON у рядку {line_num}: {e}")
                        continue
            
            # Об'єднання з базовими категоріями
            self.categories = {**self.base_categories, **loaded_categories}
            self._compile_patterns()
            
            self.logger.info(f"✅ Завантажено {len(loaded_categories)} категорій з файлу")
            self.logger.info(f"📊 Всього категорій: {len(self.categories)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Помилка завантаження категорій: {e}")
            self.categories = self.base_categories.copy()
            self._compile_patterns()
            return False
    
    def _generate_category_id(self, category_name: str) -> str:
        """Генерація ID категорії з назви"""
        # Транслітерація та очищення
        transliteration = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
            'ь': '', 'ы': 'y', 'ъ': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
            'і': 'i', 'ї': 'yi', 'є': 'ye'
        }
        
        # Нормалізація назви
        name_lower = category_name.lower()
        result = ""
        
        for char in name_lower:
            if char in transliteration:
                result += transliteration[char]
            elif char.isalnum():
                result += char
            elif char in [' ', '-', '_', ',', '.']:
                result += '_'
        
        # Очищення та обмеження довжини
        result = re.sub(r'_+', '_', result).strip('_')
        result = result[:50]  # Обмеження довжини
        
        return result if result else 'unknown_category'
    
    def _extract_keywords_from_name(self, category_name: str) -> List[str]:
        """Витягування ключових слів з назви категорії"""
        # Видалення стоп-слів
        stop_words = {'на', 'та', 'і', 'в', 'з', 'по', 'для', 'або', 'а', 'the', 'and', 'or', 'of', 'to'}
        
        # Розділення на слова
        words = re.findall(r'\b\w+\b', category_name.lower())
        
        # Фільтрація стоп-слів і коротких слів
        keywords = [word for word in words if len(word) >= 3 and word not in stop_words]
        
        return keywords
    
    def _generate_regex_patterns(self, category_name: str) -> List[str]:
        """Генерація regex патернів для категорії"""
        keywords = self._extract_keywords_from_name(category_name)
        patterns = []
        
        for keyword in keywords:
            # Патерн для точного збігу та варіацій
            pattern = rf"\b{re.escape(keyword)}\w*\b"
            patterns.append(pattern)
        
        return patterns
    
    def _compile_patterns(self):
        """Компіляція regex патернів для швидкого пошуку"""
        self.category_patterns = {}
        
        for cat_id, cat_data in self.categories.items():
            if not cat_data.get('active', True):
                continue
            
            patterns = cat_data.get('regex_patterns', [])
            if patterns:
                try:
                    # Об'єднання всіх патернів категорії
                    combined_pattern = '|'.join(patterns)
                    self.category_patterns[cat_id] = re.compile(combined_pattern, re.IGNORECASE)
                except re.error as e:
                    self.logger.warning(f"Помилка компіляції патерну для {cat_id}: {e}")
    
    def categorize_item(self, item_name: str, min_confidence: float = 0.3) -> List[Tuple[str, float]]:
        """
        Категоризація товару з оцінкою впевненості
        
        Returns:
            List[Tuple[category_id, confidence_score]]
        """
        if not item_name or not isinstance(item_name, str):
            return [("unknown", 0.0)]
        
        item_lower = item_name.lower()
        category_scores = {}
        
        # 1. Пошук за regex патернами
        for cat_id, pattern in self.category_patterns.items():
            matches = pattern.findall(item_lower)
            if matches:
                # Розрахунок впевненості на основі кількості та довжини збігів
                match_score = len(matches) * 0.3 + sum(len(match) for match in matches) * 0.01
                category_scores[cat_id] = min(match_score, 1.0)
        
        # 2. Пошук за ключовими словами
        for cat_id, cat_data in self.categories.items():
            keywords = cat_data.get('keywords', [])
            if not keywords:
                continue
            
            keyword_matches = 0
            total_keyword_length = 0
            
            for keyword in keywords:
                if keyword in item_lower:
                    keyword_matches += 1
                    total_keyword_length += len(keyword)
            
            if keyword_matches > 0:
                # Розрахунок впевненості
                keyword_score = (keyword_matches / len(keywords)) * 0.7 + (total_keyword_length / len(item_lower)) * 0.3
                
                # Комбінування з наявним score
                if cat_id in category_scores:
                    category_scores[cat_id] = max(category_scores[cat_id], keyword_score)
                else:
                    category_scores[cat_id] = keyword_score
        
        # Фільтрація за мінімальною впевненістю та сортування
        filtered_categories = [
            (cat_id, score) for cat_id, score in category_scores.items() 
            if score >= min_confidence
        ]
        
        filtered_categories.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_categories if filtered_categories else [("unknown", 0.0)]
    
    def get_primary_category(self, item_name: str) -> str:
        """Отримання основної категорії товару"""
        categories = self.categorize_item(item_name)
        return categories[0][0] if categories else "unknown"
    
    def process_historical_data(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """
        Обробка історичних даних для оновлення статистики категорій
        """
        self.logger.info("🔄 Оновлення статистики категорій...")
        
        # Скидання статистики
        self.category_competition = defaultdict(lambda: {
            'total_tenders': 0,
            'total_suppliers': 0,
            'total_positions': 0,
            'unique_suppliers': set(),
            'tender_supplier_counts': [],
            'prices': [],
            'win_rates_by_supplier': defaultdict(lambda: {'won': 0, 'total': 0}),
            'monthly_activity': defaultdict(int),
            'supplier_market_shares': defaultdict(int)
        })
        
        # Групування даних по тендерах
        tenders_by_category = defaultdict(lambda: defaultdict(list))
        
        for item in historical_data:
            item_name = item.get('F_ITEMNAME', '')
            if not item_name:
                continue
            
            # Категоризація товару
            categories = self.categorize_item(item_name)
            primary_category = categories[0][0] if categories else "unknown"
            
            tender_num = item.get('F_TENDERNUMBER', '')
            edrpou = item.get('EDRPOU', '')
            
            if tender_num and edrpou:
                tenders_by_category[primary_category][tender_num].append(item)
        
        # Аналіз кожної категорії
        results = {
            'processed_categories': 0,
            'new_categories': 0,
            'updated_competition_metrics': 0
        }
        
        for category_id, category_tenders in tenders_by_category.items():
            self._analyze_category_competition(category_id, category_tenders)
            results['processed_categories'] += 1
        
        # Фіналізація розрахунків
        self._finalize_competition_metrics()
        
        results['updated_competition_metrics'] = len(self.category_competition)
        
        self.logger.info(f"✅ Оновлено метрики для {results['processed_categories']} категорій")
        
        return results
    
    def _analyze_category_competition(self, category_id: str, category_tenders: Dict[str, List[Dict]]):
        """Аналіз конкуренції в категорії"""
        stats = self.category_competition[category_id]
        
        stats['total_tenders'] = len(category_tenders)
        all_suppliers = set()
        tender_supplier_counts = []
        all_prices = []
        supplier_wins = defaultdict(int)
        supplier_participations = defaultdict(int)
        
        for tender_num, tender_items in category_tenders.items():
            # Унікальні постачальники в тендері
            tender_suppliers = set(item.get('EDRPOU') for item in tender_items if item.get('EDRPOU'))
            tender_supplier_counts.append(len(tender_suppliers))
            all_suppliers.update(tender_suppliers)
            
            # Аналіз цін та переможців
            for item in tender_items:
                edrpou = item.get('EDRPOU')
                if edrpou:
                    supplier_participations[edrpou] += 1
                    
                    # Ціна
                    price = item.get('ITEM_BUDGET')
                    if price:
                        try:
                            price_val = float(price)
                            all_prices.append(price_val)
                        except:
                            pass
                    
                    # Перемога
                    if item.get('WON'):
                        supplier_wins[edrpou] += 1
        
        # Розрахунок метрик
        stats['total_suppliers'] = len(all_suppliers)
        stats['unique_suppliers'] = all_suppliers
        stats['tender_supplier_counts'] = tender_supplier_counts
        stats['prices'] = all_prices
        stats['supplier_participations'] = supplier_participations
        stats['supplier_wins'] = supplier_wins
        
        # Середня кількість постачальників на тендер
        if tender_supplier_counts:
            stats['avg_suppliers_per_tender'] = np.mean(tender_supplier_counts)
        
        # Інтенсивність конкуренції (0-1)
        if stats['total_tenders'] > 0:
            # Базується на середній кількості учасників
            avg_suppliers = stats['avg_suppliers_per_tender']
            stats['competition_intensity'] = min(avg_suppliers / 10, 1.0)  # Нормалізація до 1
        
        # Волатильність цін
        if len(all_prices) > 1:
            stats['price_volatility'] = np.std(all_prices) / np.mean(all_prices) if np.mean(all_prices) > 0 else 0
        
        # Бар'єр входу (складність участі)
        # Базується на мінімальній ціні та кількості учасників
        if all_prices:
            min_price = min(all_prices)
            avg_participants = stats['avg_suppliers_per_tender']
            stats['entry_barrier'] = min((min_price / 100000) * 0.3 + (1 / max(avg_participants, 1)) * 0.7, 1.0)
    
    def _finalize_competition_metrics(self):
        """Фіналізація розрахунків метрик конкуренції"""
        for category_id, stats in self.category_competition.items():
            # Концентрація ринку (Індекс Херфіндаля-Хіршмана спрощений)
            if stats['supplier_participations']:
                total_participations = sum(stats['supplier_participations'].values())
                market_shares = [count / total_participations for count in stats['supplier_participations'].values()]
                hhi = sum(share ** 2 for share in market_shares)
                stats['market_concentration'] = hhi
            
            # Середній win rate
            if stats['supplier_participations']:
                win_rates = []
                for supplier, participations in stats['supplier_participations'].items():
                    wins = stats['supplier_wins'].get(supplier, 0)
                    win_rate = wins / participations if participations > 0 else 0
                    win_rates.append(win_rate)
                
                stats['avg_win_rate'] = np.mean(win_rates) if win_rates else 0
            
            # Топ постачальники
            if stats['supplier_wins']:
                top_suppliers = sorted(
                    stats['supplier_wins'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                stats['top_suppliers'] = top_suppliers
            
            # Очищення тимчасових даних
            for key in ['unique_suppliers', 'tender_supplier_counts', 'prices', 
                       'supplier_participations', 'supplier_wins']:
                if key in stats:
                    del stats[key]
    
    def get_category_competition_metrics(self, category_id: str) -> Dict[str, Any]:
        """Отримання метрик конкуренції для категорії"""
        if category_id not in self.category_competition:
            return {'error': 'Категорія не знайдена або немає даних'}
        
        metrics = dict(self.category_competition[category_id])
        
        # Додавання інтерпретації
        metrics['interpretation'] = self._interpret_competition_metrics(metrics)
        
        return metrics
    
    def _interpret_competition_metrics(self, metrics: Dict) -> Dict[str, str]:
        """Інтерпретація метрик конкуренції"""
        interpretation = {}
        
        # Інтенсивність конкуренції
        intensity = metrics.get('competition_intensity', 0)
        if intensity < 0.3:
            interpretation['competition'] = "Низька конкуренція"
        elif intensity < 0.6:
            interpretation['competition'] = "Помірна конкуренція"
        else:
            interpretation['competition'] = "Висока конкуренція"
        
        # Концентрація ринку
        concentration = metrics.get('market_concentration', 0)
        if concentration < 0.15:
            interpretation['market'] = "Конкурентний ринок"
        elif concentration < 0.25:
            interpretation['market'] = "Помірно концентрований ринок"
        else:
            interpretation['market'] = "Високо концентрований ринок"
        
        # Бар'єр входу
        barrier = metrics.get('entry_barrier', 0)
        if barrier < 0.3:
            interpretation['entry'] = "Низький бар'єр входу"
        elif barrier < 0.6:
            interpretation['entry'] = "Помірний бар'єр входу"
        else:
            interpretation['entry'] = "Високий бар'єр входу"
        
        return interpretation
    
    def get_category_info(self, category_id: str) -> Dict[str, Any]:
        """Отримання повної інформації про категорію"""
        if category_id not in self.categories:
            return {'error': 'Категорія не знайдена'}
        
        category_data = self.categories[category_id].copy()
        
        # Додавання статистики конкуренції
        if category_id in self.category_competition:
            category_data['competition_metrics'] = self.get_category_competition_metrics(category_id)
        
        return category_data
    
    def get_supplier_category_performance(self, edrpou: str) -> Dict[str, Dict]:
        """Аналіз ефективності постачальника по категоріях"""
        return dict(self.supplier_category_performance.get(edrpou, {}))
    
    def export_state(self) -> Dict[str, Any]:
        """Експорт стану менеджера категорій"""
        return {
            'categories': self.categories,
            'category_competition': dict(self.category_competition),
            'supplier_category_performance': dict(self.supplier_category_performance)
        }
    
    def load_state(self, state_data: Dict[str, Any]):
        """Завантаження стану менеджера категорій"""
        self.categories = state_data.get('categories', {})
        self.category_competition = defaultdict(dict, state_data.get('category_competition', {}))
        self.supplier_category_performance = defaultdict(
            lambda: defaultdict(dict), 
            state_data.get('supplier_category_performance', {})
        )
        self._compile_patterns()
    
    def add_custom_category(self, 
                          category_name: str, 
                          keywords: List[str], 
                          parent_category: Optional[str] = None) -> str:
        """Додавання нової користувацької категорії"""
        category_id = self._generate_category_id(category_name)
        
        new_category = {
            "id": category_id,
            "name": category_name,
            "active": True,
            "keywords": keywords,
            "regex_patterns": [rf"\b{re.escape(kw)}\w*\b" for kw in keywords],
            "parent_category": parent_category,
            "subcategories": [],
            "competition_level": "medium",
            "source": "custom",
            "created_at": datetime.now().isoformat()
        }
        
        self.categories[category_id] = new_category
        self._compile_patterns()
        
        self.logger.info(f"✅ Додано нову категорію: {category_name} ({category_id})")
        
        return category_id
    
    def get_all_categories(self, active_only: bool = True) -> Dict[str, Dict]:
        """Отримання всіх категорій"""
        if active_only:
            return {k: v for k, v in self.categories.items() if v.get('active', True)}
        return self.categories.copy()
    
    def get_category_hierarchy(self) -> Dict[str, List[str]]:
        """Отримання ієрархії категорій"""
        hierarchy = defaultdict(list)
        
        for cat_id, cat_data in self.categories.items():
            parent = cat_data.get('parent_category')
            if parent:
                hierarchy[parent].append(cat_id)
            else:
                hierarchy['root'].append(cat_id)
        
        return dict(hierarchy)