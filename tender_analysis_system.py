import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML та NLP бібліотеки
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib

# Qdrant для векторної бази
from qdrant_client import QdrantClient
from qdrant_client.http import models
from features.feature_extractor import FeatureExtractor

class TenderAnalysisSystem:
    """
    Головна система аналізу та прогнозування тендерів
    
    Координує роботу всіх підсистем:
    - Категоризацію та аналіз
    - Векторну базу даних
    - Прогнозування
    - Аналітику конкуренції
    """
    
    def __init__(self, 
                 model_path: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 categories_file: Optional[str] = None,
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333):
        
        # Налаштування логування
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Ініціалізація основних компонентів
        self.logger.info("🚀 Ініціалізація TenderAnalysisSystem...")
        
        # Модель для ембедингів
        self.embedding_model = SentenceTransformer(model_path)
        self.logger.info(f"✅ Завантажено модель ембедингів: {model_path}")
        
        # Завантаження категорій
        self.categories_manager = None  # Буде ініціалізовано пізніше
        self.categories_file = categories_file
        
        # Підсистеми (будуть ініціалізовані пізніше)
        self.vector_db = None
        self.predictor = None        
        self.supplier_profiler = None
        self.feature_extractor = None
        
        # Конфігурація Qdrant
        self.qdrant_config = {
            'host': qdrant_host,
            'port': qdrant_port
        }
        
        # Стан системи
        self.is_initialized = False
        self.is_trained = False
        self.last_update = None
        
        # Метрики системи
        self.system_metrics = {
            'total_tenders': 0,
            'total_suppliers': 0,
            'total_categories': 0,
            'model_performance': {},
            'last_training_date': None,
            'vector_db_size': 0
        }

    def prepare_training_data(self) -> bool:
        """Підготовка даних для навчання моделі"""
        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована")
        
        # Отримання історичних даних з векторної бази
        historical_data = self.vector_db.export_collection_data(limit=100000)
        
        # Підготовка профілів постачальників
        if hasattr(self.supplier_profiler, 'profiles') and self.supplier_profiler.profiles:
            supplier_profiles = self.supplier_profiler.profiles
        else:
            # Завантаження з файлу якщо є
            profiles_file = "supplier_profiles_COMPLETE.json"
            if Path(profiles_file).exists():
                self.supplier_profiler.load_profiles(profiles_file)
                supplier_profiles = self.supplier_profiler.profiles
            else:
                self.logger.error("Профілі постачальників не знайдено!")
                return False
        
        # Підготовка даних для predictor
        X, y = self.predictor.prepare_training_data_from_history(
            [item['payload'] for item in historical_data if 'payload' in item],
            supplier_profiles
        )
        
        self.predictor.training_data = (X, y)
        return True

    def initialize_system(self) -> bool:
        """
        Повна ініціалізація всіх підсистем
        """
        try:
            self.logger.info("🔧 Ініціалізація підсистем...")
            
            # 1. Ініціалізація менеджера категорій
            from prediction.category_manager import CategoryManager
            self.categories_manager = CategoryManager(self.categories_file)
            
            if Path("categories_map.json").exists():            
                self.categories_manager.load_category_mappings("categories_map.json")
                self.logger.info("✅ Завантажено маппінг категорій")

            # 2. Ініціалізація векторної бази
            from vector_db.vector_database import TenderVectorDB
            self.vector_db = TenderVectorDB(
                embedding_model=self.embedding_model,
                qdrant_host=self.qdrant_config['host'],
                qdrant_port=self.qdrant_config['port']
            )
            
            # 3. Ініціалізація ринкової статистики
            from profiles.market_statistics import MarketStatistics
            self.market_stats = MarketStatistics(
                category_manager=self.categories_manager
            )
            
            # Завантаження або створення статистики
            if not self.market_stats.load_statistics():
                self.logger.info("📊 Статистика не знайдена. Буде створена при першому навчанні.")
            
            # 4. Ініціалізація профайлера постачальників
            from profiles.supplier_profiler import SupplierProfiler
            self.supplier_profiler = SupplierProfiler(
                categories_manager=self.categories_manager,
                vector_db=self.vector_db
            )
            
            
            # 6. Ініціалізація системи прогнозування
            from prediction.prediction_engine import PredictionEngine
            self.predictor = PredictionEngine(
                supplier_profiler=self.supplier_profiler,
                categories_manager=self.categories_manager
            )
            
            # 7. Ініціалізація екстрактора ознак
            self.feature_extractor = FeatureExtractor(
                categories_manager=self.categories_manager
            )
            
            # Передаємо market_stats в feature_extractor
            self.feature_extractor.set_market_statistics(self.market_stats)
            
            # Передаємо feature_extractor в predictor
            self.predictor.feature_extractor = self.feature_extractor
            
            self.is_initialized = True
            self.logger.info("✅ Всі підсистеми ініціалізовано успішно")
            
            return True
            
        except Exception as e:
            # self.logger.error(f"❌ Помилка ініціалізації: {e}")
            self.logger.error(e)
            return False
        

    def update_market_statistics(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Оновлення ринкової статистики
        
        Args:
            use_cache: Якщо True, використовує all_data_cache.pkl замість бази даних
        """
        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована")
        
        self.logger.info("📊 Оновлення ринкової статистики...")
        
        if use_cache and Path("files/all_data_cache.pkl").exists():
            # Використовуємо кеш
            self.logger.info("📂 Використання all_data_cache.pkl...")
            results = self.market_stats.calculate_market_statistics_from_cache("files/all_data_cache.pkl")
        else:
            # Отримання історичних даних з бази
            historical_data = []
            offset = None
            
            while True:
                try:
                    records, next_offset = self.vector_db.client.scroll(
                        collection_name=self.vector_db.collection_name,
                        offset=offset,
                        limit=10000,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    if not records:
                        break
                        
                    for record in records:
                        if record.payload:
                            historical_data.append(record.payload)
                    
                    if not next_offset:
                        break
                    offset = next_offset
                    
                except Exception as e:
                    self.logger.error(f"Помилка завантаження: {e}")
                    break
            
            # Розрахунок статистики
            results = self.market_stats.calculate_market_statistics(historical_data)
        
        self.logger.info(f"✅ Статистика оновлена для категорій")
        
        return results
    
    
    def _process_historical_data_to_features(self, historical_data: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Обробка історичних даних у фічі та таргети
        
        Args:
            historical_data: Список словників з даними тендерів
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: X (фічі) та y (таргети)
        """
        # Перетворюємо в DataFrame для зручності
        df = pd.DataFrame(historical_data)
        
        # Діагностика даних
        self.logger.info(f"\n📊 Діагностика даних:")
        self.logger.info(f"   • Всього записів: {len(df):,}")
        self.logger.info(f"   • Колонки: {list(df.columns)}")
        
        # Перевірка наявності основних полів
        required_fields = ['EDRPOU', 'F_ITEMNAME', 'WON']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            self.logger.warning(f"⚠️ Відсутні обов'язкові поля: {missing_fields}")
            # Спробуємо знайти альтернативні назви
            field_mapping = {
                'EDRPOU': ['edrpou', 'supplier_id'],
                'F_ITEMNAME': ['item_name', 'itemname'],
                'WON': ['won', 'is_winner']
            }
            
            for missing_field in missing_fields:
                alternatives = field_mapping.get(missing_field, [])
                found_alternative = None
                
                for alt in alternatives:
                    if alt in df.columns:
                        found_alternative = alt
                        break
                
                if found_alternative:
                    df[missing_field] = df[found_alternative]
                    self.logger.info(f"   ✅ Замінено {missing_field} -> {found_alternative}")
                else:
                    self.logger.error(f"   ❌ Не знайдено альтернативи для {missing_field}")
        
        # Очищення даних
        initial_count = len(df)
        
        # Видаляємо записи без основних полів
        df = df.dropna(subset=['EDRPOU', 'F_ITEMNAME'])
        
        # Видаляємо дублікати
        df = df.drop_duplicates()
        
        cleaned_count = len(df)
        self.logger.info(f"   • Після очищення: {cleaned_count:,} записів (видалено {initial_count - cleaned_count:,})")
        
        # Витягування ознак для кожного запису
        self.logger.info("🔧 Витягування ознак...")
        
        features_list = []
        targets = []
        
        # Використовуємо tqdm для прогресу
        from tqdm import tqdm
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Обробка записів"):
            # Конвертуємо рядок DataFrame назад у словник
            item = row.to_dict()
            
            # Витягуємо ЄДРПОУ для пошуку профілю
            edrpou = item.get('EDRPOU', '')
            
            # Отримуємо профіль постачальника (якщо є)
            supplier_profile = None
            if edrpou and hasattr(self.supplier_profiler, 'profiles'):
                supplier_profile = self.supplier_profiler.profiles.get(edrpou)
                if supplier_profile and hasattr(supplier_profile, 'to_dict'):
                    supplier_profile = supplier_profile.to_dict()
            
            # Витягуємо всі ознаки
            try:
                features = self.feature_extractor.extract_features(item, supplier_profile)
                features_list.append(features)
                
                # Цільова змінна - чи виграв постачальник
                targets.append(int(item.get('WON', False)))
                
            except Exception as e:
                self.logger.debug(f"Помилка обробки запису {idx}: {e}")
                continue
        
        # Створюємо фінальні структури даних
        X = pd.DataFrame(features_list)
        y = pd.Series(targets)
        
        # Заповнення пропусків
        X = X.fillna(0)
        
        # Діагностика результатів
        self.logger.info(f"\n📊 Результати витягування ознак:")
        self.logger.info(f"   • Всього зразків: {len(X):,}")
        self.logger.info(f"   • Кількість ознак: {len(X.columns)}")
        self.logger.info(f"   • Розподіл класів:")
        self.logger.info(f"     - WON=0 (програли): {(y == 0).sum():,} ({(y == 0).sum() / len(y) * 100:.1f}%)")
        self.logger.info(f"     - WON=1 (виграли): {(y == 1).sum():,} ({(y == 1).sum() / len(y) * 100:.1f}%)")
        
        # Перевірка балансу класів
        if (y == 1).sum() < len(y) * 0.05:
            self.logger.warning("⚠️ Дуже мало позитивних прикладів (<5%). Модель може погано навчатися!")
        
        # Додаткова діагностика ознак
        self.logger.info(f"\n🔍 Топ-10 ознак:")
        feature_importance_preview = []
        for col in X.columns[:10]:
            non_zero_count = (X[col] != 0).sum()
            feature_importance_preview.append(f"   • {col}: {non_zero_count:,} не-нульових значень")
        
        for line in feature_importance_preview:
            self.logger.info(line)
        
        return X, y



    
    def prepare_training_data_from_vector_db(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Підготовка навчальних даних з векторної бази або кешу
        
        Args:
            use_cache: Якщо True, спочатку спробує завантажити з files/all_data_cache.pkl
        """
        cache_file = "files/all_data_cache.pkl"
        
        # Спробуємо завантажити з кешу якщо він існує
        if use_cache and Path(cache_file).exists():
            self.logger.info(f"📂 Завантаження даних з кешу {cache_file}...")
            
            try:
                import pickle
                
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Перевірка структури кешу та конвертація
                historical_data = []
                
                if isinstance(cached_data, dict):
                    # Структура з prf.py: {edrpou: [positions]}
                    self.logger.info(f"✅ Завантажено дані для {len(cached_data)} постачальників з кешу")
                    
                    # Перетворюємо в плоский список позицій
                    for edrpou, positions in cached_data.items():
                        for position in positions:
                            # Нормалізуємо структуру для обробки
                            normalized = {
                                'EDRPOU': position.get('edrpou', edrpou),
                                'F_TENDERNUMBER': position.get('tender_number', ''),
                                'F_INDUSTRYNAME': position.get('industry', ''),
                                'F_ITEMNAME': position.get('item_name', ''),
                                'F_TENDERNAME': position.get('tender_name', ''),
                                'F_DETAILNAME': position.get('detail_name', ''),
                                'OWNER_NAME': position.get('owner_name', ''),
                                'supp_name': position.get('supplier_name', ''),
                                'CPV': position.get('cpv', 0),
                                'ITEM_BUDGET': position.get('budget', 0),
                                'F_qty': position.get('quantity', 0),
                                'F_price': position.get('price', 0),
                                'F_TENDERCURRENCY': position.get('currency', 'UAH'),
                                'F_TENDERCURRENCYRATE': position.get('currency_rate', 1.0),
                                'WON': position.get('won', False),
                                'DATEEND': position.get('date_end', ''),
                                'EXTRACTION_DATE': position.get('extraction_date', ''),
                                'ID': position.get('original_id', '')
                            }
                            historical_data.append(normalized)
                    
                    self.logger.info(f"✅ Перетворено в {len(historical_data):,} записів з кешу")
                    
                elif isinstance(cached_data, list):
                    # Вже готовий список записів
                    historical_data = cached_data
                    self.logger.info(f"✅ Завантажено {len(historical_data):,} записів з кешу")
                    
                else:
                    self.logger.warning("⚠️ Невідомий формат кешу, переходимо до векторної бази")
                    historical_data = None
                
                # Якщо дані з кешу успішно завантажені, використовуємо їх
                if historical_data:
                    return self._process_historical_data_to_features(historical_data)
                    
            except Exception as e:
                self.logger.error(f"❌ Помилка завантаження кешу: {e}")
                self.logger.info("📊 Переходимо до завантаження з векторної бази...")
        
        # Якщо кеш недоступний або використання кешу вимкнено - завантажуємо з векторної бази
        self.logger.info("📊 Завантаження даних з векторної бази...")
        
        # Збираємо дані по постачальниках
        all_records = []
        offset = None
        batch_size = 30000
        
        while True:
            try:
                records, next_offset = self.vector_db.client.scroll(
                    collection_name=self.vector_db.collection_name,
                    offset=offset,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=False  # Вектори не потрібні для навчання
                )
                
                if not records:
                    break
                    
                # Додаємо payload кожного запису
                for record in records:
                    if record.payload:
                        all_records.append(record.payload)
                
                if not next_offset:
                    break
                offset = next_offset
                
                self.logger.info(f"   Завантажено {len(all_records):,} записів...")
                
            except Exception as e:
                self.logger.error(f"Помилка завантаження: {e}")
                break
        
        self.logger.info(f"✅ Завантажено {len(all_records):,} записів з векторної бази")
        
        return self._process_historical_data_to_features(all_records)



   
    def train_prediction_model(self, validation_split: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        """Навчання моделі прогнозування на даних з векторної бази"""
        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована")
        
        # Оновлюємо ринкову статистику перед навчанням (з кешу)
        self.logger.info("📊 Оновлення ринкової статистики перед навчанням...")
        self.update_market_statistics(use_cache=True)


        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована")
        
        # 1. Перевірка наявності даних у векторній базі
        db_size = self.vector_db.get_collection_size()
        if db_size < 1000:
            raise ValueError(f"Недостатньо даних у векторній базі: {db_size}. Мінімум 1000 записів.")
        
        self.logger.info(f"🎯 Початок навчання на {db_size:,} записах з векторної бази...")
        
        try:
            # 2. Підготовка даних безпосередньо з векторної бази
            X, y = self.prepare_training_data_from_vector_db()
            
            # 3. Збереження даних у predictor для подальшого використання
            self.predictor.training_data = (X, y)
            
            # 4. Навчання моделей
            training_results = self.predictor.train_models(
                X, y,
                test_size=validation_split,
                use_calibration=True
            )
            
            # 5. Оновлення метрик системи
            self.system_metrics['model_performance'] = training_results
            self.system_metrics['last_training_date'] = datetime.now().isoformat()
            self.system_metrics['training_samples'] = len(X)
            
            self.is_trained = True
            
            self.logger.info("✅ Модель навчена успішно")
            self.logger.info(f"📈 Ensemble AUC: {training_results.get('ensemble', {}).get('test_auc', 0):.4f}")
            
            # 6. Вивід результатів по кожній моделі
            for model_name, metrics in training_results.items():
                if model_name != 'ensemble' and isinstance(metrics, dict):
                    self.logger.info(f"   • {model_name}: AUC = {metrics.get('test_auc', 0):.4f}")
            
            return {
                'performance_metrics': training_results,
                'training_samples': len(X),
                'feature_count': len(X.columns),
                'positive_rate': (y == 1).sum() / len(y)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Помилка навчання: {e}")
            raise

    def predict_tender_outcomes(self, 
                              tender_data: List[Dict],
                              include_competition_analysis: bool = True,
                              include_similar_tenders: bool = True) -> Dict[str, Any]:
        """
        Прогнозування результатів тендерів
        
        Args:
            tender_data: Дані тендерів для прогнозування
            include_competition_analysis: Включити аналіз конкуренції
            include_similar_tenders: Включити пошук схожих тендерів
        """
        if not self.is_trained:
            raise RuntimeError("Модель не натренована. Викличте train_prediction_model()")
        
        self.logger.info(f"🔮 Прогнозування для {len(tender_data)} позицій...")
        
        results = {
            'predictions': {},
            'competition_analysis': {},
            'similar_tenders': {},
            'category_insights': {},
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            # 1. Основні прогнози
            predictions = self.predictor.predict_tender(tender_data, supplier_profiles=self.supplier_profiler.profiles)
            results['predictions'] = predictions
                       
            
            # 3. Пошук схожих тендерів (якщо потрібно)
            if include_similar_tenders:
                similar_data = {}
                for item in tender_data:
                    tender_num = item.get('F_TENDERNUMBER')
                    if tender_num:
                        similar_data[tender_num] = self.vector_db.search_similar_tenders(item)
                results['similar_tenders'] = similar_data
            
            # 4. Аналіз по категоріях
            category_insights = self._analyze_predictions_by_category(tender_data, predictions)
            results['category_insights'] = category_insights
            
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ Прогнозування завершено за {results['processing_time']:.2f} сек")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Помилка прогнозування: {e}")
            raise
    
    def get_supplier_analytics(self, 
                             edrpou: str,
                             include_competition: bool = True) -> Dict[str, Any]:
        """
        Отримання повної аналітики по постачальнику
        """
        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована")
        
        # Базовий профіль
        profile = self.supplier_profiler.get_supplier_profile(edrpou)
        if not profile:
            return {'error': 'Постачальника не знайдено'}
        
        result = {
            'profile': profile,
            'competition_metrics': {},
            'category_performance': {},
            'recommendations': []
        }
           
        
        # Аналіз по категоріях
        result['category_performance'] = self.categories_manager.get_supplier_category_performance(edrpou)
        
        # Рекомендації
        result['recommendations'] = self._generate_supplier_recommendations(profile)
        
        return result
    
    def get_category_analytics(self, category_id: str) -> Dict[str, Any]:
        """
        Аналітика по категорії
        """
        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована")
        
        return {
            'category_info': self.categories_manager.get_category_info(category_id),            
            'top_suppliers': self.supplier_profiler.get_top_suppliers_in_category(category_id),
            'market_trends': self._analyze_category_trends(category_id)
        }

    def validate_training_readiness(self) -> Tuple[bool, List[str]]:
        """Перевірка готовності до навчання"""
        issues = []
        
        # Перевірка ініціалізації
        if not self.is_initialized:
            issues.append("Система не ініціалізована")
        
        # Перевірка векторної бази
        db_size = self.vector_db.get_collection_size()
        if db_size < 1000:
            issues.append(f"Недостатньо даних у векторній базі: {db_size} (мінімум 1000)")
        
        # Перевірка профілів
        if not self.supplier_profiler.profiles:
            issues.append("Відсутні профілі постачальників")
        
        # Перевірка категорій
        if not self.categories_manager.categories:
            issues.append("Не завантажені категорії")
        
        # Перевірка feature extractor
        if not hasattr(self.predictor, 'feature_extractor'):
            issues.append("Feature extractor не ініціалізований")
        
        return len(issues) == 0, issues
    
    
    def update_incremental(self, new_data: List[Dict]) -> Dict[str, Any]:
        """
        Інкрементальне оновлення системи новими даними
        """
        self.logger.info(f"🔄 Інкрементальне оновлення з {len(new_data)} записів...")
        
        # Обробка нових даних
        processing_results = self.load_and_process_data(new_data, update_mode=True)
        
        # Часткове перенавчання моделі (якщо потрібно)
        if len(new_data) > 100:  # Переначаємо тільки при значних оновленнях
            self.logger.info("📚 Часткове переначення моделі...")
            training_results = self.train_prediction_model()
            processing_results['model_retrained'] = True
            processing_results['new_performance'] = training_results['performance_metrics']
        else:
            processing_results['model_retrained'] = False
        
        return processing_results
    
    def export_system_state(self) -> Dict[str, Any]:
        """
        Експорт поточного стану системи
        """
        return {
            'system_info': {
                'version': '1.0.0',
                'initialized': self.is_initialized,
                'trained': self.is_trained,
                'last_update': self.last_update.isoformat() if self.last_update else None
            },
            'metrics': self.system_metrics,
            'categories_count': len(self.categories_manager.categories) if self.categories_manager else 0,
            'suppliers_count': len(self.supplier_profiler.profiles) if self.supplier_profiler else 0,
            'vector_db_size': self.vector_db.get_collection_size() if self.vector_db else 0
        }
    
    def save_system(self, filepath: str):
        """
        Збереження повного стану системи
        """
        self.logger.info(f"💾 Збереження системи до {filepath}...")
        
        system_data = {
            'system_state': self.export_system_state(),
            'categories_manager': self.categories_manager.export_state() if self.categories_manager else None,
            'supplier_profiler': self.supplier_profiler.export_state() if self.supplier_profiler else None,
            'predictor': self.predictor.export_state() if self.predictor else None            
        }
        
        joblib.dump(system_data, filepath)
        self.logger.info("✅ Систему збережено")
    
    def load_system(self, filepath: str):
        """
        Завантаження збереженого стану системи
        """
        self.logger.info(f"📂 Завантаження системи з {filepath}...")
        
        system_data = joblib.load(filepath)
        
        # Відновлення стану підсистем
        if system_data.get('categories_manager'):
            self.categories_manager.load_state(system_data['categories_manager'])
        
        if system_data.get('supplier_profiler'):
            self.supplier_profiler.load_state(system_data['supplier_profiler'])
        
        if system_data.get('predictor'):
            self.predictor.load_state(system_data['predictor'])                
        
        # Відновлення основного стану
        system_state = system_data.get('system_state', {})
        system_info = system_state.get('system_info', {})
        self.is_initialized = system_info.get('initialized', False)
        self.is_trained = system_info.get('trained', False)
        self.system_metrics = system_info.get('metrics', {})
        
        last_update_str = system_info.get('last_update')
        if last_update_str:
            self.last_update = datetime.fromisoformat(last_update_str)
        
        self.logger.info("✅ Систему завантажено")
    

    # Приватні допоміжні методи
    
    def _update_system_metrics(self, data: List[Dict]):
        """Оновлення системних метрик"""
        self.system_metrics['total_tenders'] = len(set(item.get('F_TENDERNUMBER') for item in data))
        self.system_metrics['total_suppliers'] = len(set(item.get('EDRPOU') for item in data))
        self.system_metrics['vector_db_size'] = len(data)
    
    def _analyze_predictions_by_category(self, tender_data: List[Dict], predictions: Dict) -> Dict:
        """Аналіз прогнозів по категоріях"""
        category_stats = defaultdict(lambda: {'count': 0, 'avg_probability': 0, 'probabilities': []})
        
        for item in tender_data:
            tender_num = item.get('F_TENDERNUMBER')
            if tender_num and tender_num in predictions:
                categories = self.categories_manager.categorize_item(item.get('F_ITEMNAME', ''))
                prob = predictions[tender_num]
                
                for category in categories:
                    category_stats[category]['count'] += 1
                    category_stats[category]['probabilities'].append(prob)
        
        # Розрахунок середніх значень
        for category, stats in category_stats.items():
            if stats['probabilities']:
                stats['avg_probability'] = np.mean(stats['probabilities'])
                stats['std_probability'] = np.std(stats['probabilities'])
                del stats['probabilities']  # Видаляємо сирі дані
        
        return dict(category_stats)
    
    def _generate_supplier_recommendations(self, profile: Dict) -> List[str]:
        """Генерація рекомендацій для постачальника"""
        recommendations = []
        
        win_rate = profile.get('position_win_rate', 0)
        if win_rate < 0.3:
            recommendations.append("Розглянути участь у менш конкурентних категоріях")
        
        if len(profile.get('item_categories', {})) < 3:
            recommendations.append("Розширити асортимент послуг/товарів")
        
        if profile.get('recent_performance', 0) < profile.get('position_win_rate', 0):
            recommendations.append("Проаналізувати причини зниження ефективності")
        
        return recommendations
    
    def _analyze_category_trends(self, category_id: str) -> Dict:
        """Аналіз трендів у категорії"""
        # Заглушка для аналізу трендів
        return {
            'trend': 'stable',
            'growth_rate': 0.0,
            'seasonal_patterns': []
        }