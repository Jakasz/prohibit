#!/usr/bin/env python3
"""
Standalone Vector Database Loader for Tenders
Ізольований швидкий завантажувач даних у Qdrant без залежностей від основної системи

Автор: Assistant
Версія: 1.0.0
"""

import json
import hashlib
import logging
import sys
import os
import re
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

# Sentence transformers
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


@dataclass
class LoaderConfig:
    """Конфігурація завантажувача"""
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "tender_vectors"
    
    # Модель
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768
    
    # Процесінг
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Оптимізація
    use_cache: bool = True
    cache_size: int = 50000
    validate_data: bool = True
    skip_existing: bool = True
    
    # Логування
    log_level: str = "INFO"
    log_file: str = "vector_loader.log"
    
    # Обмеження
    max_text_length: int = 512
    min_text_length: int = 2


class TenderDataValidator:
    """Валідатор даних тендерів"""
    
    # Обов'язкові поля для різних сценаріїв
    REQUIRED_FIELDS_SETS = [
        # Мінімальний набір 1: тендер + постачальник
        ['F_TENDERNUMBER', 'EDRPOU'],
        # Мінімальний набір 2: тендер + товар
        ['F_TENDERNUMBER', 'F_ITEMNAME'],
        # Мінімальний набір 3: постачальник + товар
        ['EDRPOU', 'F_ITEMNAME']
    ]
    
    # Всі можливі поля для збереження
    ALL_FIELDS = {
        # Ідентифікатори
        'ID', 'F_TENDERNUMBER', 'EDRPOU', 
        
        # Назви та описи
        'F_TENDERNAME', 'F_ITEMNAME', 'F_DETAILNAME', 'F_INDUSTRYNAME',
        'OWNER_NAME', 'supp_name',
        
        # Фінанси
        'ITEM_BUDGET', 'F_qty', 'F_price', 'F_TENDERCURRENCY', 'F_TENDERCURRENCYRATE',
        
        # Класифікація
        'CPV', 'F_codeUA',
        
        # Дати
        'DATEEND', 'EXTRACTION_DATE',
        
        # Результат
        'WON'
    }
    
    @staticmethod
    def validate_record(record: Dict) -> Tuple[bool, str]:
        """Валідація одного запису"""
        # Перевірка чи є хоча б один набір обов'язкових полів
        for field_set in TenderDataValidator.REQUIRED_FIELDS_SETS:
            if all(record.get(field) for field in field_set):
                return True, "OK"
        
        # Якщо жоден набір не підходить
        missing = []
        for field in ['F_TENDERNUMBER', 'EDRPOU', 'F_ITEMNAME']:
            if not record.get(field):
                missing.append(field)
        
        return False, f"Missing: {', '.join(missing)}"
    
    @staticmethod
    def extract_text_for_embedding(record: Dict) -> str:
        """Витягування тексту для створення ембедингу"""
        text_parts = []
        
        # Пріоритетні текстові поля
        priority_fields = ['F_ITEMNAME', 'F_TENDERNAME', 'F_DETAILNAME']
        for field in priority_fields:
            value = record.get(field, '').strip()
            if value:
                text_parts.append(value)
        
        # Додаткові поля для контексту
        if record.get('F_INDUSTRYNAME'):
            text_parts.append(f"галузь: {record['F_INDUSTRYNAME']}")
        
        if record.get('OWNER_NAME'):
            text_parts.append(f"замовник: {record['OWNER_NAME']}")
        
        # Якщо тексту мало, додаємо технічну інформацію
        if len(' '.join(text_parts)) < 20:
            if record.get('CPV'):
                text_parts.append(f"CPV: {record['CPV']}")
            if record.get('F_TENDERNUMBER'):
                text_parts.append(f"тендер: {record['F_TENDERNUMBER']}")
        
        return ' '.join(text_parts).strip()


class VectorDBLoader:
    """Основний клас завантажувача"""
    
    def __init__(self, config: LoaderConfig):
        self.config = config
        self.setup_logging()
        
        # Ініціалізація клієнтів
        self.client = None
        self.embedding_model = None
        
        # Кеші
        self.embedding_cache = {}
        self.hash_cache = set()
        
        # Статистика
        self.stats = defaultdict(int)
        
    def setup_logging(self):
        """Налаштування логування"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Файловий handler
        file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Консольний handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        
        # Налаштування логера
        self.logger = logging.getLogger('VectorDBLoader')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Відключення логів від бібліотек
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
    
    def initialize(self):
        """Ініціалізація компонентів"""
        self.logger.info("="*60)
        self.logger.info("🚀 Ініціалізація Vector Database Loader")
        self.logger.info("="*60)
        
        # 1. Підключення до Qdrant
        self.logger.info(f"📡 Підключення до Qdrant {self.config.qdrant_host}:{self.config.qdrant_port}")
        try:
            self.client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                timeout=30
            )
            # Тест підключення
            self.client.get_collections()
            self.logger.info("✅ Підключення до Qdrant успішне")
        except Exception as e:
            self.logger.error(f"❌ Помилка підключення до Qdrant: {e}")
            raise
        
        # 2. Завантаження моделі
        self.logger.info(f"🤖 Завантаження моделі {self.config.model_name}")
        try:
            self.embedding_model = SentenceTransformer(self.config.model_name)
            # Відключаємо прогрес-бар
            self.embedding_model.show_progress_bar = False
            self.logger.info("✅ Модель завантажена")
        except Exception as e:
            self.logger.error(f"❌ Помилка завантаження моделі: {e}")
            raise
        
        # 3. Ініціалізація колекції
        self._init_collection()
    
    def _init_collection(self):
        """Створення або перевірка колекції БЕЗ індексації"""
        collections = [col.name for col in self.client.get_collections().collections]
        
        if self.config.collection_name in collections:
            info = self.client.get_collection(self.config.collection_name)
            self.logger.info(f"📊 Колекція існує: {info.points_count:,} записів")
            
            if self.config.skip_existing:
                # Завантажуємо існуючі хеші
                self.logger.info("📥 Завантаження існуючих хешів...")
                self._load_existing_hashes()
        else:
            self.logger.info(f"🔨 Створення нової колекції '{self.config.collection_name}'")
            
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.embedding_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True  # Для великих об'ємів
                ),
                # Оптимізація для швидкого завантаження
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=4,
                    max_segment_size=2000000,
                    memmap_threshold=100000,
                    indexing_threshold=999999999,  # Відключаємо індексацію
                    flush_interval_sec=300,
                    max_optimization_threads=2
                ),
                # HNSW відключений для швидкості
                hnsw_config=models.HnswConfigDiff(
                    m=0,
                    ef_construct=100,
                    full_scan_threshold=999999999,
                    max_indexing_threads=0,
                    on_disk=True
                ),
                shard_number=1,
                replication_factor=1
            )
            
            self.logger.info("✅ Колекція створена")
    
    def _load_existing_hashes(self):
        """Завантаження хешів існуючих записів"""
        offset = None
        while True:
            try:
                records, offset = self.client.scroll(
                    collection_name=self.config.collection_name,
                    offset=offset,
                    limit=10000,
                    with_payload=["content_hash"],
                    with_vectors=False
                )
                
                for record in records:
                    if record.payload and 'content_hash' in record.payload:
                        self.hash_cache.add(record.payload['content_hash'])
                
                if not offset:
                    break
                    
            except Exception as e:
                self.logger.error(f"Помилка завантаження хешів: {e}")
                break
        
        self.logger.info(f"✅ Завантажено {len(self.hash_cache):,} хешів")
    
    def generate_content_hash(self, record: Dict) -> str:
        """Генерація унікального хешу для запису"""
        # Використовуємо всі важливі поля для унікальності
        key_parts = []
        
        # Основні ідентифікатори
        for field in ['F_TENDERNUMBER', 'EDRPOU', 'F_ITEMNAME', 'OWNER_NAME', 
                     'F_INDUSTRYNAME', 'CPV', 'ITEM_BUDGET', 'WON', 'ID']:
            value = record.get(field, '')
            key_parts.append(str(value))
        
        key_string = '|'.join(key_parts)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def create_embedding(self, text: str) -> Optional[np.ndarray]:
        """Створення ембедингу з кешуванням"""
        if not text:
            return None
        
        # Перевірка кешу
        if self.config.use_cache and text in self.embedding_cache:
            self.stats['cache_hits'] += 1
            return self.embedding_cache[text]
        
        # Обмеження довжини
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]
        
        try:
            # Створення ембедингу
            embedding = self.embedding_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Валідація
            if embedding is None or embedding.shape[0] != self.config.embedding_dim:
                return None
            
            # Кешування
            if self.config.use_cache and len(self.embedding_cache) < self.config.cache_size:
                self.embedding_cache[text] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.debug(f"Помилка створення ембедингу: {e}")
            return None
    
    def prepare_point(self, record: Dict, point_id: Optional[int] = None) -> Optional[models.PointStruct]:
        """Підготовка точки для вставки"""
        # 1. Валідація
        if self.config.validate_data:
            is_valid, reason = TenderDataValidator.validate_record(record)
            if not is_valid:
                self.stats['validation_failed'] += 1
                self.logger.warning(f"Пропущено (невалідний): {json.dumps(record, ensure_ascii=False)} | Причина: {reason}")
                return None
        
        # 2. Перевірка дублікатів
        content_hash = self.generate_content_hash(record)
        if self.config.skip_existing and content_hash in self.hash_cache:
            self.stats['duplicates_skipped'] += 1
            self.logger.warning(f"Пропущено (дубль): {json.dumps(record, ensure_ascii=False)} | Причина: дублікат по content_hash")
            return None
        
        # 3. Створення тексту для ембедингу
        embedding_text = TenderDataValidator.extract_text_for_embedding(record)
        if len(embedding_text) < self.config.min_text_length:
            self.stats['text_too_short'] += 1
            self.logger.warning(f"Пропущено (короткий текст): {json.dumps(record, ensure_ascii=False)} | Причина: короткий текст для ембедингу")
            return None
        
        # 4. Створення ембедингу
        embedding = self.create_embedding(embedding_text)
        if embedding is None:
            self.stats['embedding_failed'] += 1
            self.logger.warning(f"Пропущено (embedding_failed): {json.dumps(record, ensure_ascii=False)} | Причина: не вдалося створити ембединг")
            return None
        
        # 5. Підготовка метаданих
        metadata = self._prepare_metadata(record, content_hash)
        
        # 6. Генерація ID
        if point_id is None:
            # Використовуємо частину хешу як ID
            point_id = int(content_hash[:15], 16) % (2**53)  # JavaScript safe integer
        
        # 7. Створення точки
        try:
            point = models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=metadata
            )
            
            # Додаємо хеш до кешу
            self.hash_cache.add(content_hash)
            
            return point
            
        except Exception as e:
            self.logger.debug(f"Помилка створення точки: {e}")
            self.stats['point_creation_failed'] += 1
            return None
    
    def _prepare_metadata(self, record: Dict, content_hash: str) -> Dict[str, Any]:
        """Підготовка метаданих для збереження"""
        metadata = {
            # Системні поля
            'content_hash': content_hash,
            'indexed_at': datetime.now().isoformat(),
            
            # Основні ідентифікатори
            'tender_number': record.get('F_TENDERNUMBER', ''),
            'edrpou': record.get('EDRPOU', ''),
            'original_id': str(record.get('ID', '')),
            
            # Назви
            'item_name': record.get('F_ITEMNAME', ''),
            'tender_name': record.get('F_TENDERNAME', ''),
            'detail_name': record.get('F_DETAILNAME', ''),
            
            # Учасники
            'owner_name': record.get('OWNER_NAME', ''),
            'supplier_name': record.get('supp_name', ''),
            
            # Класифікація
            'industry': record.get('F_INDUSTRYNAME', ''),
            'cpv': self._safe_int(record.get('CPV')),
            'code_ua': record.get('F_codeUA', ''),
            
            # Фінанси
            'budget': self._safe_float(record.get('ITEM_BUDGET')),
            'quantity': self._safe_float(record.get('F_qty')),
            'price': self._safe_float(record.get('F_price')),
            'currency': record.get('F_TENDERCURRENCY', 'UAH'),
            'currency_rate': self._safe_float(record.get('F_TENDERCURRENCYRATE'), 1.0),
            
            # Дати
            'date_end': record.get('DATEEND', ''),
            'extraction_date': record.get('EXTRACTION_DATE', ''),
            
            # Результат
            'won': bool(record.get('WON', False))
        }
        
        # Видалення пустих значень для економії місця
        metadata = {k: v for k, v in metadata.items() if v != '' and v is not None}
        
        return metadata
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Безпечне перетворення в float"""
        if value is None:
            return default
        try:
            if isinstance(value, str):
                value = value.replace(',', '.').strip()
            return float(value)
        except:
            return default
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Безпечне перетворення в int"""
        if value is None:
            return default
        try:
            return int(value)
        except:
            return default
    
    def upsert_batch(self, points: List[models.PointStruct], batch_num: int) -> int:
        """Вставка батчу з повторними спробами"""
        if not points:
            return 0
        
        for attempt in range(self.config.max_retries):
            try:
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                    wait=True  # Чекаємо підтвердження
                )
                
                self.logger.debug(f"✅ Батч {batch_num}: вставлено {len(points)} точок")
                return len(points)
                
            except Exception as e:
                self.logger.warning(f"Помилка батчу {batch_num}, спроба {attempt + 1}: {e}")
                
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    self.logger.error(f"❌ Не вдалося вставити батч {batch_num}")
                    self.stats['failed_batches'] += 1
                    return 0
        
        return 0
    
    def load_file(self, file_path: str, max_records: Optional[int] = None) -> Dict[str, Any]:
        """Завантаження даних з файлу"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не знайдено: {file_path}")
        
        self.logger.info(f"\n📂 Завантаження файлу: {file_path}")
        self.logger.info(f"📊 Розмір: {file_path.stat().st_size / (1024**3):.2f} GB")
        
        # Скидання статистики для файлу
        file_stats = defaultdict(int)
        start_time = datetime.now()
        
        # Лічильник рядків
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        self.logger.info(f"📝 Всього рядків: {total_lines:,}")
        
        # Обробка файлу
        points_buffer = []
        batch_num = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=min(total_lines, max_records or total_lines), 
                       desc="Обробка", unit="записів")
            
            for line_num, line in enumerate(f, 1):
                if max_records and line_num > max_records:
                    break
                
                file_stats['total_lines'] += 1
                
                try:
                    # Парсинг JSON
                    record = json.loads(line.strip())
                    file_stats['parsed_ok'] += 1
                    
                    # Підготовка точки
                    point = self.prepare_point(record)
                    if point:
                        points_buffer.append(point)
                        file_stats['points_prepared'] += 1
                    else:
                        file_stats['points_skipped'] += 1
                        # Додаємо логування для пропущених записів
                        self.logger.warning(f"Пропущено рядок (points_skipped): {json.dumps(record, ensure_ascii=False)}")
                    
                    # Вставка батчу
                    if len(points_buffer) >= self.config.batch_size:
                        batch_num += 1
                        inserted = self.upsert_batch(points_buffer, batch_num)
                        file_stats['points_inserted'] += inserted
                        file_stats['batches_sent'] += 1
                        
                        points_buffer.clear()
                        
                        # Оновлення прогресу
                        if batch_num % 10 == 0:
                            elapsed = (datetime.now() - start_time).total_seconds()
                            speed = file_stats['points_inserted'] / elapsed if elapsed > 0 else 0
                            pbar.set_postfix({
                                'inserted': f"{file_stats['points_inserted']:,}",
                                'speed': f"{speed:.0f}/s"
                            })
                    
                except json.JSONDecodeError as e:
                    file_stats['json_errors'] += 1
                    self.logger.debug(f"JSON помилка в рядку {line_num}: {e}")
                except Exception as e:
                    self.logger.warning(f"Інша помилка в рядку {line_num}: {e}")
                    file_stats['other_errors'] += 1
                    self.logger.debug(f"Помилка в рядку {line_num}: {e}")
                
                pbar.update(1)
                
                # Очищення пам'яті
                if line_num % 50000 == 0:
                    gc.collect()
        
        # Останній батч
        if points_buffer:
            batch_num += 1
            inserted = self.upsert_batch(points_buffer, batch_num)
            file_stats['points_inserted'] += inserted
            file_stats['batches_sent'] += 1
        
        pbar.close()
        
        # Фінальна статистика
        file_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        file_stats['final_db_size'] = self.client.get_collection(self.config.collection_name).points_count
        
        # Оновлення глобальної статистики
        for key, value in file_stats.items():
            self.stats[key] += value
        
        return dict(file_stats)
    
    def print_report(self, file_stats: Dict[str, Any]):
        """Вивід звіту по файлу"""
        print("\n" + "="*60)
        print("📊 ЗВІТ ЗАВАНТАЖЕННЯ")
        print("="*60)
        
        print(f"\n📈 Обробка:")
        print(f"   • Прочитано рядків: {file_stats.get('total_lines', 0):,}")
        print(f"   • Розпарсено JSON: {file_stats.get('parsed_ok', 0):,}")
        print(f"   • Підготовлено точок: {file_stats.get('points_prepared', 0):,}")
        print(f"   • Пропущено (дублікати/невалідні): {file_stats.get('points_skipped', 0):,}")
        
        print(f"\n💾 Завантаження:")
        print(f"   • Вставлено точок: {file_stats.get('points_inserted', 0):,}")
        print(f"   • Відправлено батчів: {file_stats.get('batches_sent', 0):,}")
        print(f"   • Розмір БД: {file_stats.get('final_db_size', 0):,}")
        
        print(f"\n⚠️ Помилки:")
        print(f"   • JSON помилки: {file_stats.get('json_errors', 0):,}")
        print(f"   • Інші помилки: {file_stats.get('other_errors', 0):,}")
        
        print(f"\n⏱️ Продуктивність:")
        processing_time = file_stats.get('processing_time', 0)
        print(f"   • Час обробки: {processing_time:.1f} сек")
        
        points_inserted = file_stats.get('points_inserted', 0)
        if processing_time > 0 and points_inserted > 0:
            speed = points_inserted / processing_time
            print(f"   • Швидкість: {speed:.0f} записів/сек")
        
        # Ефективність
        total_lines = file_stats.get('total_lines', 0)
        if total_lines > 0 and points_inserted > 0:
            efficiency = points_inserted / total_lines * 100
            print(f"   • Ефективність: {efficiency:.1f}%")
        
        print("="*60)
    
    def verify_collection(self) -> Dict[str, Any]:
        """Перевірка стану колекції"""
        try:
            info = self.client.get_collection(self.config.collection_name)
            
            # Тестовий пошук
            test_vector = [0.1] * self.config.embedding_dim
            start = time.time()
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=test_vector,
                limit=1
            )
            search_time = time.time() - start
            
            return {
                'status': 'OK',
                'points_count': info.points_count,
                'segments_count': info.segments_count if hasattr(info, 'segments_count') else 0,
                'search_time_ms': search_time * 1000,
                'collection_status': info.status
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def close(self):
        """Закриття з'єднань та очищення"""
        self.logger.info("\n🏁 Завершення роботи")
        
        # Фінальна перевірка
        verification = self.verify_collection()
        self.logger.info(f"📊 Фінальний стан БД: {verification}")
        
        # Збереження детальної статистики
        stats_file = f"loader_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': self.config.__dict__,
                'stats': dict(self.stats),
                'verification': verification,
                'timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Статистика збережена в {stats_file}")
        
        # Очищення
        self.embedding_cache.clear()
        self.hash_cache.clear()
        gc.collect()


def main():
    """Основна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Standalone Vector Database Loader for Tenders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:

1. Завантаження одного файлу:
   python vector_loader.py data.jsonl

2. Завантаження кількох файлів:
   python vector_loader.py file1.jsonl file2.jsonl file3.jsonl

3. Тестове завантаження (перші 10000 записів):
   python vector_loader.py --test data.jsonl

4. Завантаження з custom параметрами:
   python vector_loader.py --batch-size 2000 --collection my_tenders data.jsonl

5. Повне перезавантаження (видалити існуючу колекцію):
   python vector_loader.py --recreate data.jsonl
        """
    )
    
    # Аргументи
    parser.add_argument('files', nargs='+', help='JSONL файли для завантаження')
    parser.add_argument('--host', default='localhost', help='Qdrant host')
    parser.add_argument('--port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--collection', default='tender_vectors', help='Назва колекції')
    parser.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 
                       help='Модель для ембедингів')
    
    # Параметри обробки
    parser.add_argument('--batch-size', type=int, default=1000, help='Розмір батчу')
    parser.add_argument('--max-records', type=int, help='Максимум записів для обробки')
    parser.add_argument('--test', action='store_true', help='Тестовий режим (10000 записів)')
    
    # Опції
    parser.add_argument('--no-cache', action='store_true', help='Не використовувати кеш ембедингів')
    parser.add_argument('--no-validation', action='store_true', help='Пропустити валідацію даних')
    parser.add_argument('--force', action='store_true', help='Ігнорувати існуючі записи')
    parser.add_argument('--recreate', action='store_true', help='Видалити та створити колекцію заново')
    
    # Логування
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Рівень логування')
    parser.add_argument('--log-file', default='vector_loader.log', help='Файл логів')
    
    args = parser.parse_args()
    
    # Створення конфігурації
    config = LoaderConfig(
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection,
        model_name=args.model,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
        validate_data=not args.no_validation,
        skip_existing=not args.force,
        log_level=args.log_level,
        log_file=args.log_file
    )
    
    # Тестовий режим
    max_records = args.max_records
    if args.test:
        max_records = 20000
        print("🧪 ТЕСТОВИЙ РЕЖИМ: обробка перших 10,000 записів")
    
    # Ініціалізація завантажувача
    loader = VectorDBLoader(config)
    
    try:
        # Ініціалізація
        loader.initialize()
        
        # Видалення колекції якщо потрібно
        if args.recreate:
            response = input(f"\n⚠️  Видалити колекцію '{config.collection_name}'? (y/n): ")
            if response.lower() == 'y':
                try:
                    loader.client.delete_collection(config.collection_name)
                    loader.logger.info(f"🗑️ Колекція '{config.collection_name}' видалена")
                    # Перестворюємо
                    loader._init_collection()
                except Exception as e:
                    loader.logger.warning(f"Помилка видалення: {e}")
        
        # Обробка файлів
        total_stats = defaultdict(int)
        
        for file_path in args.files:
            print(f"\n{'='*60}")
            print(f"📁 Файл: {file_path}")
            print(f"{'='*60}")
            
            try:
                file_stats = loader.load_file(file_path, max_records)
                loader.print_report(file_stats)
                
                # Агрегація статистики
                for key, value in file_stats.items():
                    if isinstance(value, (int, float)):
                        total_stats[key] = total_stats.get(key, 0) + value
                
            except Exception as e:
                loader.logger.error(f"❌ Помилка обробки файлу {file_path}: {e}")
                import traceback
                loader.logger.debug(traceback.format_exc())
        
        # Фінальний звіт
        if len(args.files) > 1:
            print("\n" + "="*60)
            print("📊 ЗАГАЛЬНА СТАТИСТИКА")
            print("="*60)
            print(f"\n📁 Оброблено файлів: {len(args.files)}")
            print(f"📝 Всього рядків: {total_stats.get('total_lines', 0):,}")
            print(f"✅ Вставлено записів: {total_stats.get('points_inserted', 0):,}")
            print(f"⏱️ Загальний час: {total_stats.get('processing_time', 0):.1f} сек")
            
            processing_time = total_stats.get('processing_time', 0)
            points_inserted = total_stats.get('points_inserted', 0)
            if processing_time > 0 and points_inserted > 0:
                total_speed = points_inserted / processing_time
                print(f"🚀 Середня швидкість: {total_speed:.0f} записів/сек")
        
    except KeyboardInterrupt:
        loader.logger.warning("\n⚠️ Перервано користувачем")
    except Exception as e:
        loader.logger.error(f"❌ Критична помилка: {e}")
        import traceback
        loader.logger.debug(traceback.format_exc())
    finally:
        loader.close()


if __name__ == "__main__":
    main()