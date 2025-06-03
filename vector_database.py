import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import hashlib

# Qdrant для векторної бази
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


class TenderVectorDB:
    """
    Векторна база даних для тендерів з підтримкою оновлень
    
    Функції:
    - Зберігання та пошук тендерів за семантичною схожістю
    - Інкрементальні оновлення без дублювання
    - Фільтрація за метаданими
    - Аналіз схожих тендерів
    - Кластерний аналіз
    """
    
    def __init__(self, 
                 embedding_model,
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "tender_offers_history"):
        
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Підключення до Qdrant
        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.logger.info(f"✅ Підключено до Qdrant: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            self.logger.error(f"❌ Помилка підключення до Qdrant: {e}")
            raise
        
        # Ініціалізація колекції
        self._init_collection()
        
        # Кеш для уникнення повторних обчислень
        self.embedding_cache = {}
        self.hash_cache = {}  # Для відстеження унікальності записів
        
        # Статистика
        self.stats = {
            'total_indexed': 0,
            'total_updated': 0,
            'total_searches': 0,
            'last_index_time': None,
            'last_update_time': None
        }
    
    def _init_collection(self):
        """Ініціалізація або створення колекції Qdrant"""
        try:
            # Перевірка існування колекції
            collection_info = self.client.get_collection(self.collection_name)
            self.logger.info(f"✅ Колекція '{self.collection_name}' вже існує")
            
            # Отримання поточного розміру
            self.stats['total_indexed'] = collection_info.points_count
            
        except (ResponseHandlingException, UnexpectedResponse):
            # Створення нової колекції
            self.logger.info(f"🔧 Створення нової колекції '{self.collection_name}'...")
            
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,  # Розмір для paraphrase-multilingual-mpnet-base-v2
                        distance=models.Distance.COSINE
                    ),
                    # Налаштування для оптимізації
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=20000,
                        memmap_threshold=50000,
                        indexing_threshold=20000,
                        flush_interval_sec=10,
                        max_optimization_threads=2
                    ),
                    # Налаштування реплікації та шардинга
                    shard_number=1,
                    replication_factor=1
                )
                
                # Створення індексів для швидкого пошуку
                self._create_payload_indexes()
                
                self.logger.info(f"✅ Колекція '{self.collection_name}' створена успішно")
                
            except Exception as e:
                self.logger.error(f"❌ Помилка створення колекції: {e}")
                raise
    
    def _create_payload_indexes(self):
        """Створення індексів для payload полів"""
        indexes_to_create = [
            ("tender_number", models.PayloadSchemaType.KEYWORD),
            ("edrpou", models.PayloadSchemaType.KEYWORD),
            ("industry", models.PayloadSchemaType.KEYWORD),
            ("primary_category", models.PayloadSchemaType.KEYWORD),
            ("cpv", models.PayloadSchemaType.INTEGER),
            ("budget", models.PayloadSchemaType.FLOAT),
            ("won", models.PayloadSchemaType.BOOL),
            ("date_end", models.PayloadSchemaType.KEYWORD),
            ("created_at", models.PayloadSchemaType.DATETIME)
        ]
        
        for field_name, field_type in indexes_to_create:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
            except Exception as e:
                # Індекс вже може існувати
                self.logger.debug(f"Індекс для {field_name}: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Препроцесинг тексту для ембедингів"""
        if not text or not isinstance(text, str):
            return ""
        
        # Нормалізація
        text = text.lower().strip()
        
        # Видалення зайвих символів, але збереження змістовних
        text = re.sub(r'[^\w\s\-\.\(\)]', ' ', text)
        
        # Нормалізація одиниць виміру
        units_mapping = {
            r'\bшт\.?\b': 'штук',
            r'\bкг\.?\b': 'кілограм', 
            r'\bг\.?\b': 'грам',
            r'\bл\.?\b': 'літр',
            r'\bм\.?\b': 'метр',
            r'\bсм\.?\b': 'сантиметр',
            r'\bмм\.?\b': 'міліметр'
        }
        
        for pattern, replacement in units_mapping.items():
            text = re.sub(pattern, replacement, text)
        
        # Видалення множинних пробілів
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _generate_content_hash(self, item: Dict) -> str:
        """Генерація хешу для унікальної ідентифікації записів"""
        # Використовуємо ключові поля для створення унікального хешу
        key_fields = [
            item.get('F_TENDERNUMBER', ''),
            item.get('EDRPOU', ''),
            item.get('F_ITEMNAME', ''),
            str(item.get('ITEM_BUDGET', '')),
            item.get('DATEEND', '')
        ]
        
        content_str = '|'.join(str(field) for field in key_fields)
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Створення ембедингу з кешуванням"""
        # Перевірка кешу
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Створення нового ембедингу
        processed_text = self._preprocess_text(text)
        if not processed_text:
            # Порожній вектор для порожнього тексту
            embedding = np.zeros(768)
        else:
            embedding = self.embedding_model.encode(processed_text)
        
        # Збереження в кеш (обмежуємо розмір кешу)
        if len(self.embedding_cache) < 10000:
            self.embedding_cache[text] = embedding
        
        return embedding
    
    def _prepare_point(self, item: Dict, point_id: Optional[int] = None) -> models.PointStruct:
        """Підготовка точки для індексації в Qdrant"""
        # Створення ембедингу
        item_name = item.get('F_ITEMNAME', '')
        embedding = self._create_embedding(item_name)
        
        # Підготовка метаданих
        metadata = {
            'tender_number': item.get('F_TENDERNUMBER', ''),
            'supplier_name': item.get('supp_name', ''),
            'edrpou': item.get('EDRPOU', ''),
            'item_name': item_name,
            'industry': item.get('F_INDUSTRYNAME', ''),
            'cpv': int(item.get('CPV', 0)) if item.get('CPV') else 0,
            'budget': float(item.get('ITEM_BUDGET', 0)) if item.get('ITEM_BUDGET') else 0.0,
            'quantity': float(item.get('F_qty', 0)) if item.get('F_qty') else 0.0,
            'price': float(item.get('F_price', 0)) if item.get('F_price') else 0.0,
            'won': bool(item.get('WON', False)),
            'date_end': item.get('DATEEND', ''),
            'created_at': datetime.now().isoformat(),
            'content_hash': self._generate_content_hash(item)
        }
        
        # Додавання категорій (якщо є CategoryManager)
        if hasattr(self, 'category_manager') and self.category_manager:
            categories = self.category_manager.categorize_item(item_name)
            metadata['primary_category'] = categories[0][0] if categories else 'unknown'
            metadata['all_categories'] = [cat[0] for cat in categories[:3]]  # Топ-3 категорії
            metadata['category_confidence'] = categories[0][1] if categories else 0.0
        
        # Використання переданого ID або генерація нового
        if point_id is None:
            point_id = int(hashlib.md5(metadata['content_hash'].encode()).hexdigest()[:8], 16)
        
        return models.PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=metadata
        )
    
    def index_tenders(self, 
                     historical_data: List[Dict], 
                     update_mode: bool = False,
                     batch_size: int = 100) -> Dict[str, Any]:
        """
        Індексація тендерів у векторній базі
        
        Args:
            historical_data: Список тендерів для індексації
            update_mode: True для оновлення, False для повної переініціалізації
            batch_size: Розмір батчу для обробки
        """
        self.logger.info(f"🔄 Індексація {len(historical_data)} записів (update_mode: {update_mode})")
        
        start_time = datetime.now()
        results = {
            'total_processed': len(historical_data),
            'indexed_count': 0,
            'updated_count': 0,
            'skipped_count': 0,
            'error_count': 0,
            'processing_time': 0
        }
        
        # Очищення колекції якщо не update_mode
        if not update_mode:
            try:
                self.client.delete_collection(self.collection_name)
                self._init_collection()
                self.logger.info("🗑️ Колекція очищена для повної переініціалізації")
            except Exception as e:
                self.logger.warning(f"Помилка очищення колекції: {e}")
        
        # Отримання існуючих хешів для уникнення дублювання
        existing_hashes = set()
        if update_mode:
            existing_hashes = self._get_existing_hashes()
            self.logger.info(f"📋 Знайдено {len(existing_hashes)} існуючих записів")
        
        # Обробка батчами
        points_to_upsert = []
        
        for i, item in enumerate(historical_data):
            try:
                # Перевірка унікальності
                content_hash = self._generate_content_hash(item)
                
                if update_mode and content_hash in existing_hashes:
                    results['skipped_count'] += 1
                    continue
                
                # Створення точки
                point = self._prepare_point(item)
                points_to_upsert.append(point)
                
                # Виконання батчу
                if len(points_to_upsert) >= batch_size:
                    success_count = self._upsert_batch(points_to_upsert)
                    results['indexed_count'] += success_count
                    results['error_count'] += len(points_to_upsert) - success_count
                    points_to_upsert = []
                
                # Прогрес
                if (i + 1) % 1000 == 0:
                    self.logger.info(f"📈 Оброблено {i + 1}/{len(historical_data)} записів")
                    
            except Exception as e:
                self.logger.error(f"❌ Помилка обробки запису {i}: {e}")
                results['error_count'] += 1
                continue
        
        # Обробка останнього батчу
        if points_to_upsert:
            success_count = self._upsert_batch(points_to_upsert)
            results['indexed_count'] += success_count
            results['error_count'] += len(points_to_upsert) - success_count
        
        # Фіналізація
        results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        # Оновлення статистики
        self.stats['total_indexed'] = self.get_collection_size()
        self.stats['last_index_time'] = datetime.now().isoformat()
        
        self.logger.info(f"✅ Індексація завершена за {results['processing_time']:.2f} сек")
        self.logger.info(f"📊 Проіндексовано: {results['indexed_count']}, "
                        f"Пропущено: {results['skipped_count']}, "
                        f"Помилок: {results['error_count']}")
        
        return results
    
    def _get_existing_hashes(self) -> set:
        """Отримання існуючих хешів для уникнення дублювання"""
        try:
            # Отримуємо всі точки з content_hash
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Максимальний ліміт
                with_payload=["content_hash"]
            )
            
            hashes = set()
            points = scroll_result[0]
            
            for point in points:
                if point.payload and 'content_hash' in point.payload:
                    hashes.add(point.payload['content_hash'])
            
            # Якщо є ще точки, продовжуємо скролинг
            next_page_offset = scroll_result[1]
            while next_page_offset:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=next_page_offset,
                    limit=10000,
                    with_payload=["content_hash"]
                )
                
                points = scroll_result[0]
                for point in points:
                    if point.payload and 'content_hash' in point.payload:
                        hashes.add(point.payload['content_hash'])
                
                next_page_offset = scroll_result[1]
            
            return hashes
            
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання існуючих хешів: {e}")
            return set()
    
    def _upsert_batch(self, points: List[models.PointStruct]) -> int:
        """Вставка батчу точок"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return len(points)
            
        except Exception as e:
            self.logger.error(f"❌ Помилка вставки батчу: {e}")
            return 0
    
    def search_similar_tenders(self, 
                             query_item: Dict, 
                             limit: int = 10,
                             similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Пошук схожих тендерів за семантичною схожістю
        """
        self.stats['total_searches'] += 1
        
        item_name = query_item.get('F_ITEMNAME', '')
        if not item_name:
            return []
        
        try:
            # Створення ембедингу для запиту
            query_embedding = self._create_embedding(item_name)
            
            # Підготовка фільтрів (опціонально)
            query_filter = None
            
            # Фільтр по індустрії
            industry = query_item.get('F_INDUSTRYNAME')
            if industry:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="industry",
                            match=models.MatchValue(value=industry)
                        )
                    ]
                )
            
            # Пошук
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=limit * 2,  # Беремо більше для фільтрації
                with_payload=True,
                score_threshold=similarity_threshold
            )
            
            # Форматування результатів
            similar_tenders = []
            for result in search_results:
                if len(similar_tenders) >= limit:
                    break
                
                # Уникаємо повторень того ж тендеру
                if (result.payload.get('tender_number') == query_item.get('F_TENDERNUMBER') and
                    result.payload.get('edrpou') == query_item.get('EDRPOU')):
                    continue
                
                similar_tender = {
                    'similarity_score': float(result.score),
                    'tender_number': result.payload.get('tender_number', ''),
                    'supplier_name': result.payload.get('supplier_name', ''),
                    'edrpou': result.payload.get('edrpou', ''),
                    'item_name': result.payload.get('item_name', ''),
                    'industry': result.payload.get('industry', ''),
                    'category': result.payload.get('primary_category', 'unknown'),
                    'budget': result.payload.get('budget', 0),
                    'won': result.payload.get('won', False),
                    'date_end': result.payload.get('date_end', ''),
                    'cpv': result.payload.get('cpv', 0)
                }
                
                similar_tenders.append(similar_tender)
            
            return similar_tenders
            
        except Exception as e:
            self.logger.error(f"❌ Помилка пошуку схожих тендерів: {e}")
            return []
    
    def search_by_text(self, 
                      query: str, 
                      filters: Optional[Dict] = None,
                      limit: int = 20) -> List[Dict]:
        """
        Пошук тендерів за текстовим запитом
        """
        self.stats['total_searches'] += 1
        
        if not query:
            return []
        
        try:
            # Створення ембедингу для запиту
            query_embedding = self._create_embedding(query)
            
            # Підготовка фільтрів
            query_filter = None
            if filters:
                filter_conditions = []
                
                # Фільтр по категорії
                if 'category' in filters:
                    filter_conditions.append(
                        models.FieldCondition(
                            key="primary_category",
                            match=models.MatchValue(value=filters['category'])
                        )
                    )
                
                # Фільтр по індустрії
                if 'industry' in filters:
                    filter_conditions.append(
                        models.FieldCondition(
                            key="industry",
                            match=models.MatchValue(value=filters['industry'])
                        )
                    )
                
                # Фільтр по бюджету
                if 'min_budget' in filters or 'max_budget' in filters:
                    range_condition = {}
                    if 'min_budget' in filters:
                        range_condition['gte'] = filters['min_budget']
                    if 'max_budget' in filters:
                        range_condition['lte'] = filters['max_budget']
                    
                    filter_conditions.append(
                        models.FieldCondition(
                            key="budget",
                            range=models.Range(**range_condition)
                        )
                    )
                
                # Фільтр по переможцях
                if 'won_only' in filters and filters['won_only']:
                    filter_conditions.append(
                        models.FieldCondition(
                            key="won",
                            match=models.MatchValue(value=True)
                        )
                    )
                
                # Фільтр по датах
                if 'date_from' in filters or 'date_to' in filters:
                    # Додаткова логіка для дат
                    pass
                
                if filter_conditions:
                    query_filter = models.Filter(must=filter_conditions)
            
            # Пошук
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            
            # Форматування результатів
            results = []
            for result in search_results:
                tender_result = {
                    'score': float(result.score),
                    'tender_number': result.payload.get('tender_number', ''),
                    'supplier_name': result.payload.get('supplier_name', ''),
                    'edrpou': result.payload.get('edrpou', ''),
                    'item_name': result.payload.get('item_name', ''),
                    'industry': result.payload.get('industry', ''),
                    'category': result.payload.get('primary_category', 'unknown'),
                    'budget': result.payload.get('budget', 0),
                    'won': result.payload.get('won', False),
                    'date_end': result.payload.get('date_end', ''),
                    'cpv': result.payload.get('cpv', 0)
                }
                results.append(tender_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Помилка текстового пошуку: {e}")
            return []
    
    def get_collection_size(self) -> int:
        """Отримання розміру колекції"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання розміру колекції: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Отримання статистики колекції"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'points_count': collection_info.points_count,
                'segments_count': len(collection_info.segments) if collection_info.segments else 0,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value,
                'index_stats': self.stats,
                'status': collection_info.status.value
            }
            
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання статистики: {e}")
            return {'error': str(e)}
    
    def delete_tender_records(self, tender_number: str) -> bool:
        """Видалення всіх записів конкретного тендера"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tender_number",
                                match=models.MatchValue(value=tender_number)
                            )
                        ]
                    )
                )
            )
            
            self.logger.info(f"🗑️ Видалено записи тендера: {tender_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Помилка видалення тендера {tender_number}: {e}")
            return False
    
    def update_tender_records(self, tender_data: List[Dict]) -> Dict[str, int]:
        """Оновлення записів конкретного тендера"""
        if not tender_data:
            return {'updated': 0, 'errors': 0}
        
        # Отримуємо номер тендера
        tender_number = tender_data[0].get('F_TENDERNUMBER')
        if not tender_number:
            return {'updated': 0, 'errors': 1}
        
        try:
            # Видаляємо старі записи
            self.delete_tender_records(tender_number)
            
            # Додаємо нові записи
            results = self.index_tenders(tender_data, update_mode=True)
            
            return {
                'updated': results['indexed_count'],
                'errors': results['error_count']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Помилка оновлення тендера {tender_number}: {e}")
            return {'updated': 0, 'errors': 1}
    
    def cleanup_old_records(self, days_old: int = 365) -> int:
        """Очищення старих записів"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        try:
            deleted_count = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="created_at",
                                range=models.DatetimeRange(
                                    lt=cutoff_date.isoformat()
                                )
                            )
                        ]
                    )
                )
            )
            
            self.logger.info(f"🧹 Видалено {deleted_count} старих записів")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"❌ Помилка очищення старих записів: {e}")
            return 0
    
    def export_collection_data(self, limit: Optional[int] = None) -> List[Dict]:
        """Експорт даних колекції"""
        try:
            all_points = []
            
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit or 10000,
                with_payload=True,
                with_vectors=False  # Не експортуємо вектори для економії пам'яті
            )
            
            points, next_page_offset = scroll_result
            all_points.extend([{
                'id': point.id,
                'payload': point.payload
            } for point in points])
            
            # Продовжуємо скролинг якщо потрібно
            while next_page_offset and (not limit or len(all_points) < limit):
                remaining = limit - len(all_points) if limit else 10000
                
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=next_page_offset,
                    limit=min(remaining, 10000),
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_page_offset = scroll_result
                all_points.extend([{
                    'id': point.id,
                    'payload': point.payload
                } for point in points])
            
            return all_points
            
        except Exception as e:
            self.logger.error(f"❌ Помилка експорту даних: {e}")
            return []
    
    def get_database_health(self) -> Dict[str, Any]:
        """Перевірка здоров'я бази даних"""
        health_status = {
            'status': 'unknown',
            'collection_exists': False,
            'points_count': 0,
            'last_indexed': self.stats.get('last_index_time'),
            'search_performance': 'unknown',
            'errors': []
        }
        
        try:
            # Перевірка колекції
            collection_info = self.client.get_collection(self.collection_name)
            health_status['collection_exists'] = True
            health_status['points_count'] = collection_info.points_count
            
            # Тест пошуку
            start_time = datetime.now()
            self.client.search(
                collection_name=self.collection_name,
                query_vector=[0.1] * 768,
                limit=1
            )
            search_time = (datetime.now() - start_time).total_seconds()
            
            if search_time < 0.1:
                health_status['search_performance'] = 'excellent'
            elif search_time < 0.5:
                health_status['search_performance'] = 'good'
            else:
                health_status['search_performance'] = 'slow'
            
            health_status['status'] = 'healthy'
            
        except Exception as e:
            health_status['status'] = 'error'
            health_status['errors'].append(str(e))
            self.logger.error(f"❌ Помилка перевірки здоров'я БД: {e}")
        
        return health_status