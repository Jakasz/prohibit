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
import json


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
        
        # Ініціалізація колекції
        self._init_collection()

    def _create_minimal_payload_indexes(self):
        """Створення МІНІМАЛЬНИХ індексів тільки для критичних полів"""
        minimal_indexes = [
            ("tender_number", models.PayloadSchemaType.KEYWORD),
            ("edrpou", models.PayloadSchemaType.KEYWORD),
            ("won", models.PayloadSchemaType.BOOL),
        ]
        
        for field_name, field_type in minimal_indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                    wait=False
                )
            except Exception as e:
                self.logger.debug(f"Індекс для {field_name}: {e}")



    def _init_collection(self):        
        """Ініціалізація колекції БЕЗ індексації для швидкого завантаження"""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name in collection_names:
            collection_info = self.client.get_collection(self.collection_name)
            self.logger.info(f"✅ Колекція '{self.collection_name}' вже існує")
            self.stats['total_indexed'] = collection_info.points_count
        else:
            self.logger.info(f"🔧 Створення колекції '{self.collection_name}' БЕЗ індексації...")
            try:
                # Створюємо колекцію без HNSW індексу для швидкого завантаження
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,
                        distance=models.Distance.COSINE,
                        on_disk=True  # Зберігаємо на диску
                    ),
                    # Налаштування для швидкого завантаження БЕЗ індексації
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=4,  # Мінімум сегментів
                        max_segment_size=3000000,  # Дуже великі сегменти (5M записів)
                        memmap_threshold=100000,   # Великий поріг для memmap
                        indexing_threshold=99999999,  # ВІДКЛЮЧАЄМО автоматичну індексацію
                        flush_interval_sec=300,    # Рідше скидання на диск (5 хв)
                        max_optimization_threads=8 # Мінімум потоків
                    ),
                    # НЕ передаємо hnsw_config взагалі - буде використано дефолтні налаштування
                    # які будуть змінені при увімкненні індексації
                    shard_number=2,
                    replication_factor=1
                )
                
                # Створюємо тільки критичні індекси
                self._create_minimal_payload_indexes()
                self.logger.info(f"✅ Колекція створена БЕЗ індексації для швидкого завантаження")
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
        """Генерація короткого хешу на основі всіх полів (крім службових), null замінюються на ''"""
        # Копія item без службових полів
        exclude_fields = {'content_hash', 'created_at'}
        filtered = {k: (v if v is not None else '') for k, v in item.items() if k not in exclude_fields}
        # Серіалізуємо у json-рядок з сортуванням ключів
        key_str = json.dumps(filtered, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()[:16]  # Коротший хеш

    
    def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        """Створення ембедингу з валідацією"""
        
        # Перевірка вхідного тексту
        if not text or not isinstance(text, str):
            return None
        
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        processed_text = self._preprocess_text(text)
        if not processed_text or len(processed_text) < 2:
            return None
        
        try:
            # Відключаємо вивід при енкодингу
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                
                embedding = self.embedding_model.encode(
                    processed_text, 
                    show_progress_bar=False,
                    batch_size=1,
                    normalize_embeddings=True  # Автоматична нормалізація
                )
                
                sys.stdout = old_stdout
            
            # Валідація результату
            if embedding is None:
                return None
            
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            if embedding.shape[0] != 768:
                self.logger.debug(f"⚠️ Неправильний розмір ембедингу: {embedding.shape[0]}")
                return None
            
            # Перевірка на NaN та inf
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                self.logger.debug(f"⚠️ Ембединг містить NaN або inf")
                return None
            
            # Нормалізація якщо не була зроблена автоматично
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                self.logger.debug(f"⚠️ Нульовий вектор після нормалізації")
                return None
            
            # Кешування (обмежуємо розмір кешу)
            if len(self.embedding_cache) < 5000:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.debug(f"❌ Помилка створення ембедингу: {e}")
            return None

    def _prepare_point(self, item: Dict, point_id: Optional[int] = None) -> Optional[models.PointStruct]:
        """Підготовка точки для індексації в Qdrant з валідацією"""
        
        # ========== ВАЛІДАЦІЯ ВХІДНИХ ДАНИХ ==========
        
        # 1. Перевірка основних полів
        tender_number = item.get('F_TENDERNUMBER', '')
        edrpou = item.get('EDRPOU', '')
        item_name = item.get('F_ITEMNAME', '')
        
        if not tender_number:
            self.logger.debug(f"⚠️ Пустий F_TENDERNUMBER, пропускаємо запис")
            print (f"⚠️ Пустий F_TENDERNUMBER для тендера {tender_number}")
            return ""
        
        if not edrpou:
            self.logger.debug(f"⚠️ Пустий EDRPOU для тендера {tender_number}")
            print (f"⚠️ Пустий EDRPOU для тендера {tender_number}")
            return ""
        
        if not item_name:
            self.logger.debug(f"⚠️ Пустий F_ITEMNAME для тендера {tender_number}")
            print (f"⚠️ Пустий F_ITEMNAME для тендера {tender_number}")
            return ""
        
        # 2. Створення тексту для ембедингу
        item_name = item.get('F_ITEMNAME') or ''
        tender_name = item.get('F_TENDERNAME') or ''
        detail_name = item.get('F_DETAILNAME') or ''
        industry_name = item.get('F_INDUSTRYNAME') or ''
        combined_text = f"{item_name} {edrpou} {tender_number}{industry_name}".strip()
        
        if len(combined_text) < 3:
            self.logger.debug(f"⚠️ Надто короткий текст ({len(combined_text)} символів) для тендера {tender_number}")
            return None
        
        # 3. Створення ембедингу з валідацією
        try:
            embedding = self._create_embedding(combined_text)
            
            if embedding is None:
                self.logger.debug(f"⚠️ _create_embedding повернув None для тендера {tender_number}")
                return None
            
            if not isinstance(embedding, np.ndarray):
                self.logger.debug(f"⚠️ Ембединг не є numpy array для тендера {tender_number}")
                return None
            
            if embedding.shape[0] != 768:
                self.logger.debug(f"⚠️ Неправильний розмір ембедингу: {embedding.shape[0]} замість 768 для тендера {tender_number}")
                return None
            
            if np.isnan(embedding).any():
                self.logger.debug(f"⚠️ Ембединг містить NaN для тендера {tender_number}")
                return None
            
            if np.isinf(embedding).any():
                self.logger.debug(f"⚠️ Ембединг містить inf для тендера {tender_number}")
                return None
                
        except Exception as e:
            self.logger.debug(f"❌ Помилка створення ембедингу для тендера {tender_number}: {e}")
            return None
        
        # ========== СТВОРЕННЯ МЕТАДАНИХ ==========
        
        try:
            # Створення метаданих
            metadata = {
                # Основні ідентифікатори
                'tender_number': tender_number,
                'edrpou': edrpou,
                'owner_name': item.get('OWNER_NAME') or '',
                
                # Інформація про товар/послугу
                'item_name': item_name,
                'tender_name': tender_name,
                'detail_name': detail_name,
                
                # Постачальник
                'supplier_name': item.get('supp_name') or '',
                
                # Класифікація
                'industry': item.get('F_INDUSTRYNAME') or '',
                'cpv': int(item.get('CPV', 0)) if item.get('CPV') else 0,
                
                # Фінансові показники з валідацією
                'budget': 0.0,
                'quantity': 0.0,
                'price': 0.0,
                'currency': item.get('F_TENDERCURRENCY') or 'UAH',
                'currency_rate': 1.0,
                
                # Результат та дати
                'won': bool(item.get('WON', False)),
                'date_end': item.get('DATEEND') or '',
                'extraction_date': item.get('EXTRACTION_DATE') or '',
                
                # Системні поля
                'original_id': item.get('ID') or '',
                'created_at': datetime.now().isoformat(),
                'content_hash': self._generate_content_hash(item)
            }
            
            # Безпечне конвертування числових значень
            try:
                budget = item.get('ITEM_BUDGET')
                if budget is not None and str(budget).strip():
                    metadata['budget'] = float(budget)
            except (ValueError, TypeError):
                pass
            
            try:
                quantity = item.get('F_qty')
                if quantity is not None and str(quantity).strip():
                    metadata['quantity'] = float(quantity)
            except (ValueError, TypeError):
                pass
            
            try:
                price = item.get('F_price')
                if price is not None and str(price).strip():
                    metadata['price'] = float(price)
            except (ValueError, TypeError):
                pass
            
            try:
                rate = item.get('F_TENDERCURRENCYRATE')
                if rate is not None and str(rate).strip():
                    metadata['currency_rate'] = float(rate)
            except (ValueError, TypeError):
                pass
            
            # Створення композитного ключа для групування
            group_key_parts = [
                str(edrpou),
                str(item.get('F_INDUSTRYNAME', '')),
                str(item_name),
                str(item.get('OWNER_NAME', '')),
                str(tender_name)
            ]
            
            if item.get('CPV'):
                group_key_parts.append(str(item.get('CPV')))
            
            metadata['group_key'] = '-'.join(group_key_parts)
            
            # Категоризація (якщо доступна)
            if hasattr(self, 'category_manager') and self.category_manager:
                try:
                    categories = self.category_manager.categorize_item(item_name)
                    metadata['primary_category'] = categories[0][0] if categories else 'unknown'
                    metadata['all_categories'] = [cat[0] for cat in categories[:3]]
                    metadata['category_confidence'] = categories[0][1] if categories else 0.0
                except Exception as e:
                    self.logger.debug(f"Помилка категоризації для {tender_number}: {e}")
                    metadata['primary_category'] = 'unknown'
                    metadata['all_categories'] = []
                    metadata['category_confidence'] = 0.0
            else:
                metadata['primary_category'] = 'unknown'
                metadata['all_categories'] = []
                metadata['category_confidence'] = 0.0
            
        except Exception as e:
            self.logger.debug(f"❌ Помилка створення метаданих для тендера {tender_number}: {e}")
            return None
        
        # ========== СТВОРЕННЯ ТОЧКИ ==========
        
        try:
            # Генерація ID
            if point_id is None:
                point_id = int(hashlib.md5(metadata['content_hash'].encode()).hexdigest()[:8], 16)
            
            # Створення точки
            point = models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=metadata
            )
            
            # Фінальна валідація точки
            if not hasattr(point, 'id') or point.id is None:
                self.logger.debug(f"⚠️ Точка не має ID для тендера {tender_number}")
                return None
            
            if not hasattr(point, 'vector') or not point.vector:
                self.logger.debug(f"⚠️ Точка не має вектора для тендера {tender_number}")
                return None
            
            if len(point.vector) != 768:
                self.logger.debug(f"⚠️ Неправильний розмір вектора у точці: {len(point.vector)} для тендера {tender_number}")
                return None
            
            return point
            
        except Exception as e:
            self.logger.debug(f"❌ Помилка створення точки для тендера {tender_number}: {e}")
            return None
    
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


    def _create_embeddings_batch(self, texts: List[str], batch_size: int = 128) -> List[Optional[np.ndarray]]:
        """Batch-інференс ембеддингів для списку текстів (максимальна швидкість на GPU)"""
        results = []
        if not texts:
            return results
        try:
            # Відключаємо вивід при енкодингу
            with open(os.devnull, 'w') as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=batch_size,
                    normalize_embeddings=True
                )
                sys.stdout = old_stdout
            # Перетворюємо на список np.ndarray
            if isinstance(embeddings, np.ndarray):
                results = [embeddings[i] for i in range(embeddings.shape[0])]
            else:
                results = list(embeddings)
        except Exception as e:
            self.logger.debug(f"❌ Помилка batch-інференсу ембеддингів: {e}")
            results = [None] * len(texts)
        return results

    def index_tenders(self, 
                historical_data: List[Dict], 
                update_mode: bool = True,
                batch_size: int = 1000,
                embedding_batch_size: int = 128,
                prepare_pool_size: int = 100_000) -> Dict[str, Any]:
        """
        Індексація тендерів у векторній базі з batch-інференсом ембеддингів
        Тепер підготовка points йде блоками по prepare_pool_size, а вставка у Qdrant по batch_size
        """
        print(f"\n🔄 Індексація {len(historical_data):,} записів")
        print(f"📋 Режим: {'Оновлення' if update_mode else 'Повна переіндексація'}")
        print(f"📦 Розмір батчу (Qdrant): {batch_size}")
        print(f"⚡ Batch-інференс ембеддингів: {embedding_batch_size}")
        print(f"🧮 Пул підготовки points: {prepare_pool_size}")
        
        start_time = datetime.now()
        results = {
            'total_processed': len(historical_data),
            'indexed_count': 0,
            'updated_count': 0,
            'skipped_count': 0,
            'error_count': 0,
            'processing_time': 0,
            'batch_stats': []
        }
        
        if not update_mode:
            try:
                self.client.delete_collection(self.collection_name)
                self.logger.info("🗑️ Видалено стару колекцію")
                self._init_collection()
                self.logger.info("✅ Колекція перестворена")
            except Exception as e:
                self.logger.warning(f"Помилка очищення колекції: {e}")
                import traceback
                self.logger.debug(f"Повний traceback: {traceback.format_exc()}")


        
        existing_hashes = set()
        if update_mode:
            load_start = datetime.now()
            existing_hashes = self._get_existing_hashes()
            load_time = (datetime.now() - load_start).total_seconds()
            print(f"⏱️ Час завантаження хешів: {load_time:.1f} сек\n")
        
        new_items = []
        print("🔍 Перевірка на дублікати...")
        
        for item in historical_data:
            content_hash = self._generate_content_hash(item)
            if not update_mode or content_hash not in existing_hashes:
                new_items.append(item)
            else:
                results['skipped_count'] += 1
        
        print(f"📊 Результат перевірки:")
        print(f"   • Нових записів: {len(new_items):,}")
        print(f"   • Дублікатів: {results['skipped_count']:,}")
        
        if not new_items:
            print("\n✅ Всі записи вже є в базі")
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            return results
        
        print(f"\n🚀 Індексація {len(new_items):,} нових записів:")
        
        pbar = tqdm(
            total=len(new_items),
            desc="Індексація",
            unit="записів",
            ncols=100
        )
        
        points_to_upsert = []
        batch_num = 0
        point_creation_errors = 0
        
        # ДОДАНО: счетчики для діагностики
        points_prepared = 0
        points_added_to_batch = 0
        
        i = 0
        while i < len(new_items):
            # Формуємо підбатч для ембеддингів
            batch_items = new_items[i:i+embedding_batch_size]
            texts = []
            valid_indices = []
            for idx, item in enumerate(batch_items):
                tender_number = item.get('F_TENDERNUMBER', '')
                edrpou = item.get('EDRPOU', '')
                item_name = item.get('F_ITEMNAME', '')
                tender_name = item.get('F_TENDERNAME', '')
                detail_name = item.get('F_DETAILNAME', '')
                combined_text = f"{item_name} {tender_name} {detail_name}".strip()
                if not tender_number or not edrpou or not item_name or len(combined_text) < 3:
                    continue
                texts.append(combined_text)
                valid_indices.append(idx)
            # Batch-інференс ембеддингів
            embeddings = self._create_embeddings_batch(texts, batch_size=embedding_batch_size)
            for rel_idx, emb in enumerate(embeddings):
                item = batch_items[valid_indices[rel_idx]]
                points_prepared += 1
                if emb is None or not isinstance(emb, np.ndarray) or emb.shape[0] != 768 or np.isnan(emb).any() or np.isinf(emb).any():
                    point_creation_errors += 1
                    if point_creation_errors <= 10:
                        print(f"❌ Помилка створення ембедингу для запису {i+valid_indices[rel_idx]}")
                    continue
                # ...створення метаданих як у _prepare_point...
                try:
                    metadata = {
                        'tender_number': item.get('F_TENDERNUMBER', ''),
                        'edrpou': item.get('EDRPOU', ''),
                        'owner_name': item.get('OWNER_NAME') or '',
                        'item_name': item.get('F_ITEMNAME', ''),
                        'tender_name': item.get('F_TENDERNAME', ''),
                        'detail_name': item.get('F_DETAILNAME', ''),
                        'supplier_name': item.get('supp_name') or '',
                        'industry': item.get('F_INDUSTRYNAME') or '',
                        'cpv': int(item.get('CPV', 0)) if item.get('CPV') else 0,
                        'budget': 0.0,
                        'quantity': 0.0,
                        'price': 0.0,
                        'currency': item.get('F_TENDERCURRENCY') or 'UAH',
                        'currency_rate': 1.0,
                        'won': bool(item.get('WON', False)),
                        'date_end': item.get('DATEEND') or '',
                        'extraction_date': item.get('EXTRACTION_DATE') or '',
                        'original_id': item.get('ID') or '',
                        'created_at': datetime.now().isoformat(),
                        'content_hash': self._generate_content_hash(item)
                    }
                    try:
                        budget = item.get('ITEM_BUDGET')
                        if budget is not None and str(budget).strip():
                            metadata['budget'] = float(budget)
                    except (ValueError, TypeError):
                        pass
                    try:
                        quantity = item.get('F_qty')
                        if quantity is not None and str(quantity).strip():
                            metadata['quantity'] = float(quantity)
                    except (ValueError, TypeError):
                        pass
                    try:
                        price = item.get('F_price')
                        if price is not None and str(price).strip():
                            metadata['price'] = float(price)
                    except (ValueError, TypeError):
                        pass
                    try:
                        rate = item.get('F_TENDERCURRENCYRATE')
                        if rate is not None and str(rate).strip():
                            metadata['currency_rate'] = float(rate)
                    except (ValueError, TypeError):
                        pass
                    group_key_parts = [
                        str(metadata['edrpou']),
                        str(metadata['industry']),
                        str(metadata['item_name']),
                        str(metadata['owner_name']),
                        str(metadata['tender_name'])
                    ]
                    if item.get('CPV'):
                        group_key_parts.append(str(item.get('CPV')))
                    metadata['group_key'] = '-'.join(group_key_parts)
                    # Категоризація (якщо доступна)
                    if hasattr(self, 'category_manager') and self.category_manager:
                        try:
                            categories = self.category_manager.categorize_item(metadata['item_name'])
                            metadata['primary_category'] = categories[0][0] if categories else 'unknown'
                            metadata['all_categories'] = [cat[0] for cat in categories[:3]]
                            metadata['category_confidence'] = categories[0][1] if categories else 0.0
                        except Exception as e:
                            self.logger.debug(f"Помилка категоризації для {metadata['tender_number']}: {e}")
                            metadata['primary_category'] = 'unknown'
                            metadata['all_categories'] = []
                            metadata['category_confidence'] = 0.0
                    else:
                        metadata['primary_category'] = 'unknown'
                        metadata['all_categories'] = []
                        metadata['category_confidence'] = 0.0
                    # Генерація ID
                    point_id = int(hashlib.md5(metadata['content_hash'].encode()).hexdigest()[:8], 16)
                    point = models.PointStruct(
                        id=point_id,
                        vector=emb.tolist(),
                        payload=metadata
                    )
                    points_to_upsert.append(point)
                    points_added_to_batch += 1
                except Exception as e:
                    point_creation_errors += 1
                    if point_creation_errors <= 10:
                        print(f"❌ Помилка створення точки для запису {i+valid_indices[rel_idx]}: {e}")
                    continue
            # Якщо накопичили пул для вставки
            if len(points_to_upsert) >= prepare_pool_size:
                # Вставляємо у Qdrant по batch_size
                while len(points_to_upsert) >= batch_size:
                    batch_num += 1
                    batch_points = points_to_upsert[:batch_size]
                    print(f"\n🔀 Батч #{batch_num}: підготовлено {points_prepared} записів, у батчі {len(batch_points)} точок")
                    batch_start = datetime.now()
                    success_count = self._upsert_batch(batch_points, batch_num)
                    batch_time = (datetime.now() - batch_start).total_seconds()
                    batch_stat = {
                        'batch_num': batch_num,
                        'points_prepared': len(batch_points),
                        'points_uploaded': success_count,
                        'points_failed': len(batch_points) - success_count,
                        'batch_time': batch_time
                    }
                    results['batch_stats'].append(batch_stat)
                    print(f"✅ Батч #{batch_num}: відправлено {success_count}/{len(batch_points)} точок за {batch_time:.2f}с")
                    results['indexed_count'] += success_count
                    results['error_count'] += len(batch_points) - success_count
                    # Додаємо лог по кількості записів у базі
                    current_count = self.get_collection_size()
                    print(f"📊 Поточна кількість записів у базі після батчу #{batch_num}: {current_count}")
                    points_to_upsert = points_to_upsert[batch_size:]
                    points_added_to_batch = len(points_to_upsert)
                    if batch_num % 10 == 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        speed = results['indexed_count'] / elapsed if elapsed > 0 else 0
                        pbar.set_postfix(
                            batches=batch_num,
                            indexed=f"{results['indexed_count']:,}",
                            speed=f"{speed:.0f}/с"
                        )
            pbar.update(len(batch_items))
            i += embedding_batch_size
        # Вставка залишку
        if points_to_upsert:
            while len(points_to_upsert) > 0:
                batch_num += 1
                batch_points = points_to_upsert[:batch_size]
                print(f"\n🔀 Батч #{batch_num}: {len(batch_points)} точок (залишок)")
                batch_start = datetime.now()
                success_count = self._upsert_batch(batch_points, batch_num)
                batch_time = (datetime.now() - batch_start).total_seconds()
                batch_stat = {
                    'batch_num': batch_num,
                    'points_prepared': len(batch_points),
                    'points_uploaded': success_count,
                    'points_failed': len(batch_points) - success_count,
                    'batch_time': batch_time
                }
                results['batch_stats'].append(batch_stat)
                print(f"✅ Батч #{batch_num}: відправлено {success_count}/{len(batch_points)} точок")
                results['indexed_count'] += success_count
                results['error_count'] += len(batch_points) - success_count
                points_to_upsert = points_to_upsert[batch_size:]
        pbar.close()
        
        # Детальна статистика батчів
        if results['batch_stats']:
            print(f"\n📊 Статистика батчів:")
            total_prepared = sum(b['points_prepared'] for b in results['batch_stats'])
            total_uploaded = sum(b['points_uploaded'] for b in results['batch_stats'])
            avg_batch_size = total_prepared / len(results['batch_stats'])
            
            print(f"   • Всього батчів: {len(results['batch_stats'])}")
            print(f"   • Середній розмір батчу: {avg_batch_size:.1f}")
            print(f"   • Всього підготовлено точок: {total_prepared:,}")
            print(f"   • Всього завантажено точок: {total_uploaded:,}")
            print(f"   • Помилок створення точок: {point_creation_errors}")
            
            # Перші кілька батчів для діагностики
            print(f"\n📋 Перші 5 батчів:")
            for batch in results['batch_stats'][:5]:
                print(f"   Батч {batch['batch_num']}: {batch['points_prepared']}/{batch_size} підготовлено, "
                    f"{batch['points_uploaded']} завантажено, {batch['batch_time']:.2f}с")
        
        # Фіналізація
        results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        # Статистика
        self.stats['total_indexed'] = self.get_collection_size()
        self.stats['last_index_time'] = datetime.now().isoformat()
        
        # Фінальна статистика
        print("\n" + "="*60)
        print("✅ ІНДЕКСАЦІЯ ЗАВЕРШЕНА")
        print("="*60)
        print(f"📊 Загальна статистика:")
        print(f"   • Оброблено записів: {results['total_processed']:,}")
        print(f"   • Проіндексовано нових: {results['indexed_count']:,}")
        print(f"   • Пропущено дублікатів: {results['skipped_count']:,}")
        print(f"   • Помилок обробки: {results['error_count']:,}")
        print(f"   • Час обробки: {results['processing_time']:.1f} сек")
        
        if results['indexed_count'] > 0:
            print(f"   • Швидкість: {results['indexed_count']/results['processing_time']:.0f} записів/сек")
        
        print(f"\n📊 Стан колекції:")
        print(f"   • Всього записів у базі: {self.stats['total_indexed']:,}")
        print("="*60)
        
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
        """Спрощена версія отримання існуючих хешів без прогрес-бару"""
        try:
            hashes = set()
            offset = None
            batch_count = 0
            
            print(f"📋 Завантаження існуючих хешів...")
            
            while True:
                # Отримуємо батч записів
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    offset=offset,
                    limit=10000,
                    with_payload=["content_hash"],
                    with_vectors=False
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                    
                # Додаємо хеші
                for point in points:
                    if point.payload and 'content_hash' in point.payload:
                        hashes.add(point.payload['content_hash'])
                
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"   • Завантажено {len(hashes):,} хешів...")
                
                if not next_offset:
                    break
                    
                offset = next_offset
            
            print(f"✅ Завантажено {len(hashes):,} унікальних хешів")
            return hashes
            
        except Exception as e:            
            self.logger.error(f"❌ Помилка отримання існуючих хешів: {e}")
            return set()
    
    def _upsert_batch(self, points: List[models.PointStruct], batch_num: int = 0) -> int:
        """Швидка вставка батчу БЕЗ очікування завершення"""
        try:
            print(f"📡 Швидка відправка батчу #{batch_num} з {len(points)} точок...")
            
            # Мінімальна валідація - тільки кількість
            if not points:
                print(f"❌ Порожній батч #{batch_num}")
                return 0
            
            # 🔥 ШВИДКА відправка БЕЗ очікування
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=False,  # НЕ чекаємо завершення!
                ordering=models.WriteOrdering.WEAK  # Слабка консистентність
            )
            
            print(f"⚡ Батч #{batch_num} відправлено асинхронно")
            
            # Повертаємо кількість точок (припускаємо успіх)
            return len(points)
            
        except Exception as e:
            print(f"❌ Помилка швидкої відправки батчу #{batch_num}: {e}")
            print(f"   Тип помилки: {type(e).__name__}")
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
                
                # Уникаємо повторень того ж тендера
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

    def get_all_supplier_ids(self) -> list:
        """
        Повертає всі унікальні EDRPOU (ідентифікатори постачальників) з колекції Qdrant.
        """
        client = self.client  # ← виправлено тут
        collection = self.collection_name if hasattr(self, 'collection_name') else "tender_vectors"

        edrpou_set = set()
        offset = None

        while True:
            response = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                scroll_filter=None  # ← виправлено тут
            )
            points = response[0]
            if not points:
                break
            for point in points:
                edrpou = point.payload.get("edrpou")
                if edrpou:
                    edrpou_set.add(edrpou)
            if len(points) < 1000:
                break
            offset = points[-1].id

        return list(edrpou_set)

