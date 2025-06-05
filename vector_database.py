import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import hashlib
from tqdm import tqdm
import sys
import os
from sentence_transformers import SentenceTransformer
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
                 collection_name: str = "tender_vectors"):
        
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
        import transformers
        import datasets
        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()
        if hasattr(self.embedding_model, 'tokenizer') and self.embedding_model.tokenizer:
            self.embedding_model.tokenizer.verbose = False

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
                    collection_name=self.collection_name,  # ДОДАНО!
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE,                    
                        on_disk=True
                    ),
                    # Налаштування для оптимізації
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=4,
                        max_segment_size=100000,
                        memmap_threshold=20000,
                        indexing_threshold=50000,
                        flush_interval_sec=30,
                        max_optimization_threads=1
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,  # Зменшено з дефолтних 64
                        ef_construct=100,  # Зменшено з дефолтних 200
                        full_scan_threshold=10000,
                        max_indexing_threads=1,
                        on_disk=True,  # HNSW індекс на диску
                        payload_m=16
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=False
                        )
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
        """Створення індексів тільки для критичних полів"""
        indexes_to_create = [
            ("tender_number", models.PayloadSchemaType.KEYWORD),
            ("edrpou", models.PayloadSchemaType.KEYWORD),
            ("won", models.PayloadSchemaType.BOOL),
            ("industry", models.PayloadSchemaType.KEYWORD),
            # НЕ індексуємо F_ITEMNAME - це текстове поле, займає багато місця
        ]
        
        for field_name, field_type in indexes_to_create:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                    wait=False  # Асинхронне створення
                )
            except Exception as e:
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
        """Генерація короткого хешу"""
        key_str = f"{item.get('F_TENDERNUMBER', '')}{item.get('EDRPOU', '')}{item.get('F_ITEMNAME', '')}{item.get('F_INDUSTRYNAME', '')}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()[:16]  # Коротший хеш

    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Створення ембедингу з кешуванням та нормалізацією"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        processed_text = self._preprocess_text(text)
        if not processed_text:
            embedding = np.zeros(768)
        else:
            # Відключаємо вивід при енкодингу
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                embedding = self.embedding_model.encode(processed_text, show_progress_bar=False)
                sys.stdout = old_stdout
            
            # Нормалізація для кращої компресії
            embedding = embedding / np.linalg.norm(embedding)
        
        if len(self.embedding_cache) < 5000:
            self.embedding_cache[text] = embedding
        
        return embedding

    
    # def _create_payload_indexes(self):
    #     """Створення індексів для payload полів"""
    #     indexes_to_create = [
    #         ("tender_number", models.PayloadSchemaType.KEYWORD),
    #         ("edrpou", models.PayloadSchemaType.KEYWORD),
    #         ("owner_name", models.PayloadSchemaType.KEYWORD),  # ДОДАНО
    #         ("industry", models.PayloadSchemaType.KEYWORD),
    #         ("primary_category", models.PayloadSchemaType.KEYWORD),
    #         ("cpv", models.PayloadSchemaType.INTEGER),
    #         ("budget", models.PayloadSchemaType.FLOAT),
    #         ("won", models.PayloadSchemaType.BOOL),
    #         ("date_end", models.PayloadSchemaType.KEYWORD),
    #         ("group_key", models.PayloadSchemaType.KEYWORD),  # ДОДАНО
    #         ("created_at", models.PayloadSchemaType.DATETIME)
    #     ]


    def _prepare_point(self, item: Dict, point_id: Optional[int] = None) -> models.PointStruct:
        """Підготовка точки для індексації в Qdrant з урахуванням OWNER_NAME"""
        
        # ЗМІНЕНО: Створення комбінованого тексту для ембедингу
        combined_text = f"{item.get('F_ITEMNAME', '')} {item.get('F_TENDERNAME', '')} {item.get('F_DETAILNAME', '')}"
        embedding = self._create_embedding(combined_text)
        
        # ЗМІНЕНО: Розширені метадані
        metadata = {
            # Основні ідентифікатори
            'tender_number': item.get('F_TENDERNUMBER', ''),
            'edrpou': item.get('EDRPOU', ''),
            'owner_name': item.get('OWNER_NAME', ''),  # ДОДАНО
            
            # Інформація про товар/послугу
            'item_name': item.get('F_ITEMNAME', ''),
            'tender_name': item.get('F_TENDERNAME', ''),  # ДОДАНО
            'detail_name': item.get('F_DETAILNAME', ''),  # ДОДАНО
            
            # Постачальник
            'supplier_name': item.get('supp_name', ''),
            
            # Класифікація
            'industry': item.get('F_INDUSTRYNAME', ''),
            'cpv': int(item.get('CPV', 0)) if item.get('CPV') else 0,
            
            # Фінансові показники
            'budget': float(item.get('ITEM_BUDGET', 0)) if item.get('ITEM_BUDGET') else 0.0,
            'quantity': float(item.get('F_qty', 0)) if item.get('F_qty') else 0.0,
            'price': float(item.get('F_price', 0)) if item.get('F_price') else 0.0,
            'currency': item.get('F_TENDERCURRENCY', 'UAH'),  # ДОДАНО
            'currency_rate': float(item.get('F_TENDERCURRENCYRATE', 1.0)),  # ДОДАНО
            
            # Результат та дати
            'won': bool(item.get('WON', False)),
            'date_end': item.get('DATEEND', ''),
            'extraction_date': item.get('EXTRACTION_DATE', ''),  # ДОДАНО
            
            # Системні поля
            'original_id': item.get('ID', ''),  # ДОДАНО
            'created_at': datetime.now().isoformat(),
            'content_hash': self._generate_content_hash(item)
        }
        
        # ДОДАНО: Створення композитного ключа для групування
        group_key_parts = [
            str(item.get('EDRPOU', '')),
            str(item.get('F_INDUSTRYNAME', '')),
            str(item.get('F_ITEMNAME', '')),
            str(item.get('OWNER_NAME', '')),
            str(item.get('F_TENDERNAME', ''))
        ]
        
        if item.get('CPV'):
            group_key_parts.append(str(item.get('CPV')))
        
        metadata['group_key'] = '-'.join(group_key_parts)
        
        # Існуючий код категоризації залишається
        if hasattr(self, 'category_manager') and self.category_manager:
            categories = self.category_manager.categorize_item(item.get('F_ITEMNAME', ''))
            metadata['primary_category'] = categories[0][0] if categories else 'unknown'
            metadata['all_categories'] = [cat[0] for cat in categories[:3]]
            metadata['category_confidence'] = categories[0][1] if categories else 0.0
        
        # Використання переданого ID або генерація нового
        if point_id is None:
            point_id = int(hashlib.md5(metadata['content_hash'].encode()).hexdigest()[:8], 16)
        
        return models.PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=metadata
        )
    
    def optimize_collection(self):
        """Оптимізація колекції після індексації"""
        try:
            # Форсуємо оптимізацію сегментів
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    max_segment_size=500000,
                    memmap_threshold=100000,
                    indexing_threshold=100000
                )
            )
            self.logger.info("✅ Колекція оптимізована")
        except Exception as e:
            self.logger.error(f"❌ Помилка оптимізації: {e}")


    def index_tenders(self, 
                 historical_data: List[Dict], 
                 update_mode: bool = False,
                 batch_size: int = 1000) -> Dict[str, Any]:
        """
        Індексація тендерів у векторній базі з єдиним прогрес-баром
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
        
        # Отримання існуючих хешів
        existing_hashes = set()
        if update_mode:
            self.logger.info("📋 Завантаження існуючих хешів...")
            existing_hashes = self._get_existing_hashes()
            self.logger.info(f"✅ Знайдено {len(existing_hashes)} існуючих записів")
        
        # Підготовка прогрес-бару
        total_items = len(historical_data)
        pbar = tqdm(
            total=total_items,
            desc="🚀 Індексація",
            unit="записів",
            ncols=120,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate} записів/с]'
        )
        
        # Обробка батчами
        points_to_upsert = []
        processed_count = 0
        
        # Статистика для виводу
        stats_update_interval = 1000
        last_stats_update = 0
        
        for i, item in enumerate(historical_data):
            try:
                # Перевірка унікальності
                content_hash = self._generate_content_hash(item)
                
                if update_mode and content_hash in existing_hashes:
                    results['skipped_count'] += 1
                    pbar.set_postfix({
                        'Індексовано': results['indexed_count'],
                        'Пропущено': results['skipped_count'],
                        'Помилок': results['error_count']
                    }, refresh=False)
                    pbar.update(1)
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
                    
                    # Оновлення статистики в прогрес-барі
                    pbar.set_postfix({
                        'Індексовано': results['indexed_count'],
                        'Пропущено': results['skipped_count'],
                        'Помилок': results['error_count'],
                        'Швидкість': f"{pbar.format_dict['rate_fmt']}"
                    }, refresh=True)
                
                processed_count += 1
                pbar.update(1)
                
                # Детальна статистика кожні N записів
                if processed_count - last_stats_update >= stats_update_interval:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    eta = (total_items - processed_count) / speed if speed > 0 else 0

                    pbar.set_postfix({
                        'Індексовано': results['indexed_count'],
                        'Пропущено': results['skipped_count'],
                        'Помилок': results['error_count'],
                        'Швидкість': f"{speed:.0f}/сек",
                        'Залишилось': str(timedelta(seconds=int(eta)))
                    }, refresh=True)
                    last_stats_update = processed_count
                    
            except Exception as e:
                results['error_count'] += 1
                if results['error_count'] <= 10:  # Показуємо тільки перші 10 помилок
                    tqdm.write(f"❌ Помилка обробки запису {i}: {e}")
                pbar.update(1)
                continue
        
        # Обробка останнього батчу
        if points_to_upsert:
            pbar.set_description("🔄 Завершення останнього батчу")
            success_count = self._upsert_batch(points_to_upsert)
            results['indexed_count'] += success_count
            results['error_count'] += len(points_to_upsert) - success_count
        
        pbar.close()
        
        # Фіналізація
        results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        # Оновлення статистики
        self.stats['total_indexed'] = self.get_collection_size()
        self.stats['last_index_time'] = datetime.now().isoformat()
        
        # Фінальна статистика
        print("\n" + "="*60)
        print("✅ ІНДЕКСАЦІЯ ЗАВЕРШЕНА")
        print("="*60)
        print(f"📊 Загальна статистика:")
        print(f"   • Оброблено записів: {results['total_processed']:,}")
        print(f"   • Проіндексовано: {results['indexed_count']:,}")
        print(f"   • Пропущено дублікатів: {results['skipped_count']:,}")
        print(f"   • Помилок: {results['error_count']:,}")
        print(f"   • Час обробки: {timedelta(seconds=int(results['processing_time']))}")
        print(f"   • Середня швидкість: {results['total_processed']/results['processing_time']:.0f} записів/сек")
        print(f"   • Розмір колекції: {self.stats['total_indexed']:,} записів")
        print("="*60)
        self.optimize_collection()  
        return results

    def _format_time(self, seconds: float) -> str:
        """Форматування часу в читабельний вигляд"""
        if seconds < 60:
            return f"{seconds:.0f} сек"
        elif seconds < 3600:
            return f"{seconds/60:.1f} хв"
        else:
            return f"{seconds/3600:.1f} год"

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
    
    def search_tenders_by_group(self, group_criteria: Dict, limit: int = 20) -> List[Dict]:
        """Пошук тендерів за груповими критеріями"""
        filter_conditions = []
        
        for field, value in group_criteria.items():
            if value is not None:
                filter_conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value)
                    )
                )
        
        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )[0]
            
            return [{
                'id': point.id,
                **point.payload
            } for point in results]
        except Exception as e:
            self.logger.error(f"❌ Помилка пошуку за групою: {e}")
            return []

    def get_tender_groups(self, tender_number: str) -> Dict[str, List[Dict]]:
        """Отримання всіх груп для конкретного тендера"""
        try:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tender_number",
                        match=models.MatchValue(value=tender_number)
                    )
                ]
            )
            
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )[0]
            
            groups = defaultdict(list)
            for point in results:
                group_key = point.payload.get('group_key', 'unknown')
                groups[group_key].append(point.payload)
            
            return dict(groups)
            
        except Exception as e:
            self.logger.error(f"❌ Помилка отримання груп тендера: {e}")
            return {}

