from collections import defaultdict
import json
import logging
import os
from typing import Dict, List
from pathlib import Path

from tqdm import tqdm
# SupplierMetrics and SupplierProfile might be needed if we reconstruct profiles here,
# but the current script only caches raw data.
# from supplier_profiler import SupplierMetrics, SupplierProfile
# from tender_analysis_system import TenderAnalysisSystem # Replaced by system_provider
from system_provider import get_system, is_system_initialized

import pickle

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ProfileBuilder:    
    def __init__(self, system): # system is an instance of TenderAnalysisSystem
        self.system = system
        if not self.system or not self.system.is_initialized:
            # This check should ideally be done before creating ProfileBuilder
            logger.error("ProfileBuilder: TenderAnalysisSystem is not initialized!")
            raise ValueError("TenderAnalysisSystem must be initialized before creating ProfileBuilder.")
        self.vector_db = system.vector_db
        # self.profiler = system.supplier_profiler # Not directly used in build_and_cache
        self.logger = logger # Use the module-level logger

    def build_and_cache_supplier_data(self, cache_file_path: str = "files/all_data_cache.pkl", force_rebuild: bool = False):
        """
        Loads all tender data from the vector database, groups it by supplier EDRPOU,
        and saves it to a pickle cache file.
        This cache is intended for use by MarketStatistics and potentially other components
        that need quick access to all historical data without repeated DB queries.
        """
        self.logger.info("🚀 Початок процесу створення/оновлення кешу даних постачальників.")

        cache_path = Path(cache_file_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True) # Ensure 'files' directory exists

        if not force_rebuild and cache_path.exists():
            self.logger.info(f"📦 Знайдено існуючий кеш даних: {cache_file_path}. Завантаження з БД пропущено.")
            self.logger.info("💡 Щоб примусово перебудувати кеш, використовуйте параметр force_rebuild=True.")
            return True # Indicate cache exists and was not rebuilt

        self.logger.info(f"🔥 { 'Примусове перебудування' if force_rebuild else 'Створення нового' } кешу даних: {cache_file_path}")

        supplier_data = defaultdict(list)

        # 1. Завантаження через scroll з vector_db (який є частиною self.system)
        self.logger.info("⚡ Початок завантаження даних з векторної бази...")

        if not self.vector_db or not hasattr(self.vector_db, 'client') or not hasattr(self.vector_db, 'collection_name'):
            self.logger.error("❌ Vector DB клієнт або назва колекції не налаштовані в TenderAnalysisSystem.")
            return False

        try:
            collection_info = self.vector_db.client.get_collection(
                collection_name=self.vector_db.collection_name
            )
            total_points = collection_info.points_count
            if total_points == 0:
                self.logger.warning(f"⚠️ Векторна база даних '{self.vector_db.collection_name}' порожня. Кеш буде порожнім.")
            else:
                self.logger.info(f"📊 Всього записів для завантаження з БД: {total_points:,}")
        except Exception as e:
            self.logger.error(f"❌ Не вдалося отримати інформацію про колекцію '{self.vector_db.collection_name}': {e}")
            self.logger.error("❗ Переконайтеся, що векторна база даних існує та доступна перед створенням кешу.")
            return False

        pbar = tqdm(total=total_points, desc="Завантаження даних з БД", unit="записів")
        offset = None
        total_loaded_records = 0
        batch_size = 40000  # Оптимальний розмір батчу для scroll

        while True:
            try:
                records, next_offset = self.vector_db.client.scroll(
                    collection_name=self.vector_db.collection_name,
                    offset=offset,
                    limit=batch_size,
                    with_payload=True,  # Потрібні всі дані з payload
                    with_vectors=False  # Вектори не потрібні для цього кешу
                )
                if not records:
                    break # Немає більше записів

                for record in records:
                    if record.payload:
                        # Важливо: забезпечити однакові ключі, незалежно від регістру
                        edrpou = record.payload.get('edrpou', record.payload.get('EDRPOU'))
                        if edrpou: # Тільки якщо є ЄДРПОУ
                            supplier_data[str(edrpou)].append(record.payload) # Ключ завжди строка
                        else:
                            # self.logger.debug(f"Запис без ЄДРПОУ: {record.payload.get('id', 'N/A')}")
                            pass # Можна логувати, якщо потрібно
                    total_loaded_records +=1
                    pbar.update(1)

                if total_loaded_records % (batch_size * 5) == 0: # Логування кожні 5 батчів
                    self.logger.info(f"   Проміжно завантажено {total_loaded_records:,} записів, знайдено {len(supplier_data):,} унікальних ЄДРПОУ...")

                if not next_offset:
                    break # Кінець даних
                offset = next_offset

            except Exception as e:
                self.logger.error(f"❌ Помилка під час завантаження даних з Qdrant: {e}")
                self.logger.error("❗ Можливо, проблема з підключенням або структурою даних.")
                pbar.close()
                return False # Невдача
        pbar.close()

        self.logger.info(f"✅ Завантажено {total_loaded_records:,} записів.")
        self.logger.info(f"👥 Згруповано дані для {len(supplier_data):,} унікальних постачальників.")

        # Зберігаємо кеш даних
        self.logger.info(f"💾 Зберігання агрегованих даних у кеш-файл: {cache_file_path}...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(dict(supplier_data), f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"✅ Кеш даних постачальників успішно збережено: {cache_file_path}")
            return True # Успіх
        except Exception as e:
            self.logger.error(f"❌ Не вдалося зберегти кеш-файл: {e}")
            return False # Невдача

def run_build_cache(categories_file: str = "data/categories.jsonl",
                    qdrant_host: str = "localhost",
                    qdrant_port: int = 6333,
                    cache_file: str = "files/all_data_cache.pkl",
                    force_rebuild_cache: bool = False):
    """
    Основна функція для запуску процесу побудови кешу.
    Використовує system_provider для отримання ініціалізованої системи.
    """
    logger.info("🛠️  Запуск побудови кешу даних постачальників...")
    try:
        system = get_system(
            categories_file=categories_file,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port
        )
    except Exception as e:
        logger.exception(f"❌ Критична помилка під час отримання/ініціалізації TenderAnalysisSystem: {e}")
        return

    if not is_system_initialized() or not system.is_initialized:
        logger.error("❌ TenderAnalysisSystem не ініціалізовано належним чином. Побудова кешу неможлива.")
        return

    logger.info("✅ TenderAnalysisSystem успішно отримано/ініціалізовано.")

    builder = ProfileBuilder(system)
    success = builder.build_and_cache_supplier_data(cache_file_path=cache_file, force_rebuild=force_rebuild_cache)

    if success:
        logger.info("🎉 Побудова кешу даних постачальників завершена успішно.")
    else:
        logger.error("🔥 Побудова кешу даних постачальників завершена з помилками або була перервана.")


if __name__ == "__main__":
    # Параметри за замовчуванням для запуску як скрипт
    DEFAULT_CATEGORIES_FILE = "data/categories.jsonl" # Шлях до файлу категорій
    DEFAULT_QDRANT_HOST = "localhost"
    DEFAULT_QDRANT_PORT = 6333
    DEFAULT_CACHE_FILE = "files/all_data_cache.pkl"   # Де зберігати кеш
    FORCE_REBUILD = False # Встановіть True, щоб примусово перебудувати кеш, навіть якщо він існує

    # Приклад аргументів командного рядка (можна розширити argparse)
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == '--force-rebuild':
        logger.info("⚡ Примусове перебудування кешу активовано через аргумент командного рядка.")
        FORCE_REBUILD = True

    run_build_cache(
        categories_file=DEFAULT_CATEGORIES_FILE,
        qdrant_host=DEFAULT_QDRANT_HOST,
        qdrant_port=DEFAULT_QDRANT_PORT,
        cache_file=DEFAULT_CACHE_FILE,
        force_rebuild_cache=FORCE_REBUILD
    )