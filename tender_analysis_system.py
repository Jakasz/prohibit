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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib

# Qdrant для векторної бази
from qdrant_client import QdrantClient
from qdrant_client.http import models
from feature_extractor import FeatureExtractor

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
        self.competition_analyzer = None
        self.supplier_profiler = None
        
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
    
    def initialize_system(self) -> bool:
        """
        Повна ініціалізація всіх підсистем
        """
        try:
            self.logger.info("🔧 Ініціалізація підсистем...")
            
            # 1. Ініціалізація менеджера категорій
            from category_manager import CategoryManager  # Буде створено далі
            self.categories_manager = CategoryManager(self.categories_file)
            
            if Path("category_mappings.json").exists():            
                self.categories_manager.load_category_mappings("category_mappings.json")
                self.logger.info("✅ Завантажено маппінг категорій")


            # 2. Ініціалізація векторної бази
            from vector_database import TenderVectorDB  # Буде створено далі
            self.vector_db = TenderVectorDB(
                embedding_model=self.embedding_model,
                qdrant_host=self.qdrant_config['host'],
                qdrant_port=self.qdrant_config['port']
            )
            
            # 3. Ініціалізація профайлера постачальників
            from supplier_profiler import SupplierProfiler  # Буде створено далі
            self.supplier_profiler = SupplierProfiler(
                categories_manager=self.categories_manager,
                # embedding_model=self.embedding_model
            )
            
            # 4. Ініціалізація аналізатора конкуренції
            from competition_analyzer import CompetitionAnalyzer  # Буде створено далі
            self.competition_analyzer = CompetitionAnalyzer(
                categories_manager=self.categories_manager,
                vector_db=self.vector_db
            )
            
            # 5. Ініціалізація системи прогнозування
            from prediction_engine import PredictionEngine  # Буде створено далі
            self.predictor = PredictionEngine(
                supplier_profiler=self.supplier_profiler,
                competition_analyzer=self.competition_analyzer,
                categories_manager=self.categories_manager
            )
            # 6. Ініціалізація екстрактора ознак
            feature_extractor = FeatureExtractor(
                categories_manager=self.categories_manager,
                competition_analyzer=self.competition_analyzer
            )
            self.predictor = PredictionEngine(
                # feature_extractor=feature_extractor
                supplier_profiler = self.supplier_profiler,
                competition_analyzer = self.competition_analyzer,
                categories_manager = self.categories_manager
            )
            
            
            self.is_initialized = True
            self.logger.info("✅ Всі підсистеми ініціалізовано успішно")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Помилка ініціалізації: {e}")
            return False
    
    def load_and_process_data(self, 
                            historical_data: List[Dict],
                            update_mode: bool = False) -> Dict[str, Any]:
        """
        Завантаження та обробка історичних даних
        
        Args:
            historical_data: Список історичних записів тендерів
            update_mode: True для оновлення існуючих даних, False для повної переініціалізації
        """
        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована. Викличте initialize_system()")
        
        self.logger.info(f"📥 Обробка {len(historical_data)} записів (режим оновлення: {update_mode})")
        
        results = {
            'processed_records': len(historical_data),
            'new_suppliers': 0,
            'updated_suppliers': 0,
            'new_categories': 0,
            'vector_db_updates': 0,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            # 1. Оновлення категорій
            category_stats = self.categories_manager.process_historical_data(historical_data)
            results['new_categories'] = category_stats.get('new_categories', 0)
            
            # 2. Створення/оновлення профілів постачальників
            supplier_stats = self.supplier_profiler.build_profiles(
                historical_data, 
                update_mode=update_mode
            )
            results['new_suppliers'] = supplier_stats.get('new_profiles', 0)
            results['updated_suppliers'] = supplier_stats.get('updated_profiles', 0)
            
            # 3. Оновлення векторної бази
            vector_stats = self.vector_db.index_tenders(
                historical_data, 
                update_mode=update_mode
            )
            results['vector_db_updates'] = vector_stats.get('indexed_count', 0)
            
            # 4. Оновлення аналізу конкуренції
            self.competition_analyzer.update_competition_metrics(historical_data)
            
            # 5. Оновлення системних метрик
            self._update_system_metrics(historical_data)
            
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            self.last_update = datetime.now()
            
            self.logger.info(f"✅ Обробка завершена за {results['processing_time']:.2f} сек")
            self.logger.info(f"📊 Нові постачальники: {results['new_suppliers']}")
            self.logger.info(f"📊 Оновлені постачальники: {results['updated_suppliers']}")
            self.logger.info(f"📊 Записів у векторній БД: {results['vector_db_updates']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Помилка обробки даних: {e}")
            raise
    
    def train_prediction_model(self, 
                             validation_split: float = 0.2,
                             cross_validation_folds: int = 5) -> Dict[str, Any]:
        """
        Тренування моделі прогнозування
        """
        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована")
        
        if self.system_metrics['total_tenders'] == 0:
            raise ValueError("Немає даних для тренування. Спочатку завантажте історичні дані")
        
        self.logger.info("🎯 Початок тренування моделі прогнозування...")
        
        try:
            # Тренування моделі
            training_results = self.predictor.train_model(
                validation_split=validation_split,
                cv_folds=cross_validation_folds
            )
            
            # Оновлення метрик системи
            self.system_metrics['model_performance'] = training_results['performance_metrics']
            self.system_metrics['last_training_date'] = datetime.now().isoformat()
            
            self.is_trained = True
            
            self.logger.info("✅ Модель натренована успішно")
            self.logger.info(f"📈 AUC Score: {training_results['performance_metrics']['auc_score']:.4f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"❌ Помилка тренування: {e}")
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
            predictions = self.predictor.predict_batch(tender_data)
            results['predictions'] = predictions
            
            # 2. Аналіз конкуренції (якщо потрібно)
            if include_competition_analysis:
                competition_data = {}
                for item in tender_data:
                    tender_num = item.get('F_TENDERNUMBER')
                    if tender_num:
                        competition_data[tender_num] = self.competition_analyzer.analyze_tender_competition(item)
                results['competition_analysis'] = competition_data
            
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
    
    def search_tenders(self, 
                      query: str, 
                      filters: Optional[Dict] = None,
                      limit: int = 20) -> List[Dict]:
        """
        Пошук тендерів у векторній базі
        """
        if not self.is_initialized:
            raise RuntimeError("Система не ініціалізована")
        
        return self.vector_db.search_by_text(
            query=query,
            filters=filters,
            limit=limit
        )
    
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
        
        # Аналіз конкуренції
        if include_competition:
            result['competition_metrics'] = self.competition_analyzer.get_supplier_competition_metrics(edrpou)
        
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
            'competition_metrics': self.competition_analyzer.get_category_competition_metrics(category_id),
            'top_suppliers': self.supplier_profiler.get_top_suppliers_in_category(category_id),
            'market_trends': self._analyze_category_trends(category_id)
        }
    
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
            'predictor': self.predictor.export_state() if self.predictor else None,
            'competition_analyzer': self.competition_analyzer.export_state() if self.competition_analyzer else None
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
            
        if system_data.get('competition_analyzer'):
            self.competition_analyzer.load_state(system_data['competition_analyzer'])
        
        # Відновлення основного стану
        system_state = system_data.get('system_state', {})
        self.is_initialized = system_state.get('initialized', False)
        self.is_trained = system_state.get('trained', False)
        self.system_metrics = system_state.get('metrics', {})
        
        last_update_str = system_state.get('last_update')
        if last_update_str:
            self.last_update = datetime.fromisoformat(last_update_str)
        
        self.logger.info("✅ Систему завантажено")
    
    def create_vector_db_from_jsonl(self, jsonl_path: str, collection_name: Optional[str] = None, batch_size: int = 100) -> Dict[str, Any]:
        """
        Створення векторної бази з файлу JSONL з усіма полями
        Args:
            jsonl_path: шлях до файлу JSONL з тендерами
            collection_name: назва колекції Qdrant (опціонально)
            batch_size: розмір батчу для індексації
        Returns:
            Статистика індексації
        """
        self.logger.info(f"📂 Завантаження даних з {jsonl_path} для створення векторної бази...")
        if not Path(jsonl_path).exists():
            self.logger.error(f"❌ Файл {jsonl_path} не знайдено")
            raise FileNotFoundError(f"Файл {jsonl_path} не знайдено")

        # Завантаження даних з JSONL
        historical_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    historical_data.append(record)
                except Exception as e:
                    self.logger.warning(f"Помилка парсингу JSON у рядку {line_num}: {e}")

        self.logger.info(f"✅ Завантажено {len(historical_data)} записів з {jsonl_path}")

        # Якщо потрібно, створюємо нову колекцію з новою назвою
        if collection_name:
            from .vector_database import TenderVectorDB
            self.vector_db = TenderVectorDB(
                embedding_model=self.embedding_model,
                qdrant_host=self.qdrant_config['host'],
                qdrant_port=self.qdrant_config['port'],
                collection_name=collection_name
            )
            self.logger.info(f"🔧 Створено нову колекцію Qdrant: {collection_name}")
        elif self.vector_db is None:
            from .vector_database import TenderVectorDB
            self.vector_db = TenderVectorDB(
                embedding_model=self.embedding_model,
                qdrant_host=self.qdrant_config['host'],
                qdrant_port=self.qdrant_config['port']
            )

        # Передаємо менеджер категорій у векторну базу (для категоризації)
        if hasattr(self.vector_db, 'category_manager') and self.vector_db.category_manager is None:
            self.vector_db.category_manager = self.categories_manager

        # Індексація у векторній базі
        stats = self.vector_db.index_tenders(
            historical_data=historical_data,
            update_mode=False,
            batch_size=batch_size
        )

        self.logger.info(f"✅ Векторна база створена. Проіндексовано: {stats.get('indexed_count', 0)} записів")
        return stats
    
    def process_large_dataset(self, jsonl_path: str, batch_size: int = 1000, max_records: int = None) -> Dict[str, Any]:
        """
        Потокова обробка великого JSONL файлу
        
        Args:
            jsonl_path: шлях до JSONL файлу
            batch_size: розмір батчу для обробки
            max_records: максимальна кількість записів (для тестування)
        """
        self.logger.info(f"📂 Потокова обробка {jsonl_path}")
        
        if not Path(jsonl_path).exists():
            raise FileNotFoundError(f"Файл {jsonl_path} не знайдено")
        
        stats = {
            'total_read': 0,
            'total_indexed': 0,
            'total_errors': 0,
            'batches_processed': 0
        }
        
        batch_data = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_records and line_num > max_records:
                    break
                
                try:
                    record = json.loads(line.strip())
                    batch_data.append(record)
                    stats['total_read'] += 1
                    
                    if len(batch_data) >= batch_size:
                        # Обробка батчу
                        self._process_batch(batch_data, stats)
                        batch_data = []
                        
                        # Прогрес
                        if stats['total_read'] % 10000 == 0:
                            self.logger.info(f"Оброблено {stats['total_read']:,} записів")
                            
                except Exception as e:
                    self.logger.error(f"Помилка в рядку {line_num}: {e}")
                    stats['total_errors'] += 1
        
        # Останній батч
        if batch_data:
            self._process_batch(batch_data, stats)
        
        return stats

    def _process_batch(self, batch_data: List[Dict], stats: Dict):
        """Обробка одного батчу даних"""
        try:
            # Оновлення профілів постачальників
            suppliers_by_edrpou = defaultdict(list)
            for item in batch_data:
                edrpou = item.get('EDRPOU')
                if edrpou:
                    suppliers_by_edrpou[edrpou].append(item)
            
            # Оновлення профілів
            for edrpou, items in suppliers_by_edrpou.items():
                self.supplier_profiler.update_profile(edrpou, items)
            
            # Індексація у векторній базі
            index_results = self.vector_db.index_tenders(
                batch_data,
                update_mode=True,
                batch_size=100
            )
            
            stats['total_indexed'] += index_results['indexed_count']
            stats['total_errors'] += index_results['error_count']
            stats['batches_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Помилка обробки батчу: {e}")
            stats['total_errors'] += len(batch_data)




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