
"""
Оптимізований скрипт для створення векторної бази з підтримкою оновлення
"""

import logging
import json
from pathlib import Path
# from tender_analysis_system import TenderAnalysisSystem # Replaced by system_provider
from system_provider import get_system, is_system_initialized
from qdrant_client import QdrantClient
from tqdm import tqdm
from datetime import datetime
from qdrant_client.http import models

# Assuming TenderVectorDB is still needed for specific DB operations not covered by TenderAnalysisSystem's vector_db attribute directly
# However, TenderAnalysisSystem itself initializes and holds a TenderVectorDB instance.
# We should use system.vector_db instead of creating a separate one if possible.
# from vector_database import TenderVectorDB # This might be redundant if system.vector_db is used

# Configure logging if not already done by other modules
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


# This function seems to be a method of TenderVectorDB, not a standalone one.
# It's not used in this file's global scope. If it's a helper for TenderVectorDB, it should be there.
# For now, commenting out as it's not directly called in the main logic of create_vector_db.py
# def fast_upsert_batch(self, points, batch_num: int = 0) -> int:
#     """Швидка вставка батчу БЕЗ очікування завершення"""
#     try:
#         print(f"📡 Швидка відправка батчу #{batch_num} з {len(points)} точок...")
#
#         if not points:
#             print(f"❌ Порожній батч #{batch_num}")
#             return 0
#
#         # 🔥 ШВИДКА відправка БЕЗ очікування
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=points,
#             wait=True,  # НЕ чекаємо завершення!
#             ordering=models.WriteOrdering.WEAK  # Слабка консистентність
#         )
#
#         print(f"⚡ Батч #{batch_num} відправлено асинхронно")
#         return len(points)
#
#     except Exception as e:
#         print(f"❌ Помилка швидкої відправки батчу #{batch_num}: {e}")
#         return 0


def monitor_progress(system, start_count, interval=50000):
    """Моніторинг прогресу та розміру"""
    current_count = system.vector_db.get_collection_size()
    if current_count - start_count >= interval:
        stats = system.vector_db.get_storage_info()
        logger.info(f"""
        📊 СТАТИСТИКА:
        - Записів: {current_count:,}
        - Розмір векторів: {stats['vectors_size_gb']} ГБ
        - Розмір метаданих: {stats['payload_size_gb']} ГБ
        - Загальний розмір: {stats['estimated_total_gb']} ГБ
        """)
        return current_count
    return start_count

def process_file_with_stats(file_path, system, batch_size=1000, update_mode=True): # system is TenderAnalysisSystem
    """
    Обробка файлу з детальною статистикою
    
    Args:
        file_path: шлях до JSONL файлу
        system: екземпляр TenderAnalysisSystem
        batch_size: розмір батчу для обробки
        update_mode: True - додавання нових записів, False - повна переіндексація
    """
    logger.info(f"\n📁 Обробка файлу: {file_path}")
    file_size = Path(file_path).stat().st_size / (1024**3)
    logger.info(f"📊 Розмір файлу: {file_size:.2f} ГБ")
    
    # Підрахунок кількості рядків
    logger.info("📝 Підрахунок записів...")
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"✅ Всього записів: {total_lines:,}")
    
    # Завантаження та обробка
    data = []
    errors = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Завантаження"):
            try:
                data.append(json.loads(line.strip()))
            except Exception as e:
                errors += 1
                if errors <= 10:  # Показуємо перші 10 помилок
                    logger.warning(f"⚠️ Помилка парсингу JSON: {e}")
    
    if errors > 0:
        logger.warning(f"⚠️ Всього помилок завантаження: {errors}")
    
    # Індексація з урахуванням режиму оновлення
    logger.info(f"\n🔄 Індексація в режимі: {'оновлення' if update_mode else 'повного перестворення'}")
    # Assuming system.vector_db is an instance of TenderVectorDB or similar, and has index_tenders
    stats = system.vector_db.index_tenders(
        data,
        update_mode=update_mode,  # Передаємо режим оновлення
        batch_size=batch_size
    )
    
    return stats

def check_existing_collection(host="localhost", port=6333, collection_name="tender_vectors"):
    """
    Перевірка існування колекції та отримання інформації
    
    Returns:
        tuple: (exists: bool, info: dict)
    """
    try:
        # This client is temporary for checking existence before full system init
        client = QdrantClient(host=host, port=port)
        collection_info_model = client.get_collection(collection_name)
        
        info = {
            'exists': True,
            'points_count': collection_info_model.points_count,
            'status': collection_info_model.status,
            'vectors_size': collection_info_model.config.params.vectors.size,
            # Segments count might not be directly available or named differently depending on client version
            'segments_count': getattr(collection_info_model, 'segments_count', 0)
        }
        client.close() # Close temporary client
        return True, info
    except Exception: # Broad exception for cases where collection doesn't exist or Qdrant is down
        return False, {'exists': False}


def process_file_fast(file_path, system, batch_size=5000, update_mode=True, fast_mode=True): # system is TenderAnalysisSystem
    """ШВИДКА обробка файлу"""
    logger.info(f"⚡ ШВИДКА обробка: {file_path}")
    
    # Підрахунок записів
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"📊 Всього записів: {total_lines:,}")
    
    # Швидке завантаження та обробка
    data = []
    errors = 0
    
    logger.info("📥 Швидке завантаження в пам'ять...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Завантаження", unit="рядків"):
            try:
                data.append(json.loads(line.strip()))
            except Exception as e:
                errors += 1
                if errors <= 5:
                    logger.warning(f"⚠️ JSON помилка: {e}")
    
    logger.info(f"✅ Завантажено {len(data):,} записів (помилок: {errors})")
    
    # ШВИДКА індексація
    logger.info(f"⚡ ШВИДКА індексація з батчами по {batch_size}...")
    # Assuming system.vector_db is an instance of TenderVectorDB or similar, and has index_tenders
    stats = system.vector_db.index_tenders(
        data,
        update_mode=update_mode,
        batch_size=batch_size
        # fast_mode might be a parameter to index_tenders or handled by how system.vector_db is configured
    )
    
    return stats


def create_optimized_vector_database(
    jsonl_files: list,
    categories_file: str = "categories.jsonl", # Default from TenderAnalysisSystem
    collection_name: str = "tender_vectors",
    batch_size: int = 5000,
    update_mode: bool | None = None, # If None, will ask user if collection exists
    force_recreate: bool = False,
    fast_mode: bool = True, # This might influence how index_tenders behaves or if indexing is deferred
    qdrant_host: str = "localhost", # Default from TenderAnalysisSystem
    qdrant_port: int = 6333         # Default from TenderAnalysisSystem
):
    """
    Створює або оновлює векторну базу даних, використовуючи глобальний TenderAnalysisSystem.
    """
    
    logger.info("="*60)
    if fast_mode:
        logger.info("⚡ ШВИДКИЙ РЕЖИМ ЗАВАНТАЖЕННЯ")
        logger.info("   • Індексація може бути ВИМКНЕНА або відкладена (залежить від TenderVectorDB.index_tenders)")
        logger.info("   • Збільшений розмір батчу (якщо застосовано в index_tenders)")
    else:
        logger.info("🐌 ЗВИЧАЙНИЙ РЕЖИМ (може включати індексацію)")
    logger.info("="*60)
    
    # Перевірка існування колекції ДО ініціалізації системи (щоб не ініціалізувати, якщо скасовано)
    # Qdrant client details should ideally match what TenderAnalysisSystem will use.
    exists, collection_info = check_existing_collection(host=qdrant_host, port=qdrant_port, collection_name=collection_name)
    
    if exists and not force_recreate:
        logger.info(f"\n✅ Колекція '{collection_name}' вже існує:")
        logger.info(f"   • Записів: {collection_info.get('points_count', 'N/A'):,}")
        
        if update_mode is None: # Only ask if not specified
            choice = input("\n❓ Колекція існує. Оновити (додати нові дані) чи видалити і створити заново? (update/recreate/cancel): ").lower()
            if choice == 'recreate':
                update_mode = False # This means full recreate
            elif choice == 'update':
                update_mode = True # This means add new, skip existing
            else:
                logger.info("❌ Операція скасована користувачем.")
                return False # Indicate cancellation
                
        # Видалення якщо потрібно (force_recreate or user chose recreate)
        if not update_mode: # update_mode is False if we need to recreate
            try:
                # Use a temporary client for deletion before system init
                temp_client = QdrantClient(host=qdrant_host, port=qdrant_port)
                logger.info(f"🗑️ Спроба видалення старої колекції '{collection_name}'...")
                temp_client.delete_collection(collection_name)
                logger.info(f"🗑️ Стару колекцію '{collection_name}' видалено.")
                temp_client.close()
                exists = False # Collection no longer exists
            except Exception as e:
                # Log error but potentially continue if it's "not found"
                logger.error(f"❌ Помилка видалення колекції '{collection_name}': {e}. Можливо, її вже не було.")
                # If deletion fails for other reasons, it might be an issue.
    elif force_recreate and exists:
        logger.info(f"🔥 Примусове перестворення колекції '{collection_name}'.")
        update_mode = False # force_recreate implies not updating but starting fresh
        try:
            temp_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            logger.info(f"🗑️ Спроба видалення старої колекції '{collection_name}' для перестворення...")
            temp_client.delete_collection(collection_name)
            logger.info(f"🗑️ Стару колекцію '{collection_name}' видалено для перестворення.")
            temp_client.close()
            exists = False
        except Exception as e:
            logger.error(f"❌ Помилка видалення колекції '{collection_name}' під час примусового перестворення: {e}.")
            # Depending on strictness, might want to return False here

    # Ініціалізація системи через system_provider
    # Параметри categories_file, qdrant_host, qdrant_port будуть використані system_provider.get_system()
    # тільки при першому виклику. Якщо система вже ініціалізована, ці параметри ігноруються.
    logger.info("\n📦 Спроба отримати/ініціалізувати TenderAnalysisSystem...")
    try:
        system = get_system(categories_file=categories_file, qdrant_host=qdrant_host, qdrant_port=qdrant_port)
    except Exception as e:
        logger.exception(f"❌ Критична помилка під час отримання/ініціалізації системи: {e}")
        return False

    if not is_system_initialized() or not system.is_initialized:
        logger.error("❌ Помилка ініціалізації TenderAnalysisSystem через system_provider!")
        return False
    
    logger.info("✅ TenderAnalysisSystem отримано/ініціалізовано успішно.")

    # Переконуємося, що vector_db в system використовує правильну назву колекції.
    # TenderVectorDB.__init__ створює колекцію, якщо її немає.
    # Якщо ми видалили її вище, вона буде створена тут з правильними параметрами.
    # Якщо вона існує, TenderVectorDB підключиться до неї.
    system.vector_db.collection_name = collection_name # Ensure an explicit set, though constructor might handle it

    # Якщо колекція не існувала (або була видалена), TenderVectorDB її створить під час своєї ініціалізації (яка є частиною TenderAnalysisSystem.initialize_system)
    # Нам потрібно перевірити, чи вона справді створена, якщо ми очікуємо, що вона буде.
    # Або, якщо ми хочемо контролювати створення тут:
    if not exists: # If it didn't exist or was deleted
        logger.info(f"Спроба створити колекцію '{collection_name}' через system.vector_db (якщо ще не створено)...")
        # The vector_db instance within the system should handle its own collection creation logic.
        # This might involve calling a specific method like system.vector_db.create_collection_if_not_exists()
        # For now, we assume TenderVectorDB's constructor or an internal call in initialize_system handles this.
        # Let's ensure it's actually there after initialization.
        try:
            system.vector_db.client.get_collection(collection_name=collection_name)
            logger.info(f"Колекція '{collection_name}' існує після ініціалізації системи.")
        except Exception as e:
            logger.error(f"Колекція '{collection_name}' не знайдена навіть після ініціалізації системи: {e}")
            logger.error("Перевірте логіку створення колекції в TenderVectorDB або TenderAnalysisSystem.initialize_system().")
            return False


    # Початкова статистика
    # update_mode being True means we are adding to an existing collection (or a newly created empty one).
    # update_mode being False means we started with a fresh, empty collection.
    initial_count = 0
    if update_mode: # Only relevant if we are in "add to existing" mode
        try:
            initial_count = system.vector_db.get_collection_size()
            logger.info(f"\n📊 Початкова кількість записів (режим оновлення): {initial_count:,}")
        except Exception as e:
            logger.error(f"Не вдалося отримати початковий розмір колекції: {e}")
            initial_count = 0 # Assume 0 if error
    else: # Recreate mode
        logger.info(f"\n📊 Початкова кількість записів (режим перестворення): 0")
    
    # Обробка файлів
    total_indexed_overall = 0
    total_skipped_overall = 0
    total_errors_overall = 0
    overall_start_time = datetime.now()

    for idx, jsonl_file_path_str in enumerate(jsonl_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📁 Файл {idx}/{len(jsonl_files)}: {Path(jsonl_file_path_str).name}")
        logger.info(f"{'='*60}")
        
        if not Path(jsonl_file_path_str).exists():
            logger.error(f"❌ Файл не знайдено: {jsonl_file_path_str}")
            continue

        # Обробка файлу
        # update_mode для process_file_fast має бути True, якщо ми хочемо додавати дані
        # (тобто, index_tenders має обробляти дублікати або додавати нові).
        # Якщо update_mode для create_optimized_vector_database було False (recreate),
        # то для першого файлу це фактично не update, а початкове заповнення.
        # Для наступних файлів це завжди update до поточної колекції.
        current_file_update_mode = True # By default, index_tenders should try to update/add

        stats = process_file_fast( # Renamed from process_file_with_stats; assuming process_file_fast is the one to use
            jsonl_file_path_str,
            system, 
            batch_size=batch_size,
            update_mode=current_file_update_mode, # index_tenders handles if it's new or adding
            fast_mode=fast_mode # This might be a hint for index_tenders
        )
        
        total_indexed_overall += stats.get('indexed_count', 0)
        total_skipped_overall += stats.get('skipped_count', 0)
        total_errors_overall += stats.get('error_count', 0)

        logger.info(f"\n✅ Файл {Path(jsonl_file_path_str).name} оброблено:")
        logger.info(f"   • Проіндексовано (в цьому файлі): {stats.get('indexed_count', 0):,}")
        logger.info(f"   • Пропущено (в цьому файлі): {stats.get('skipped_count', 0):,}")
        logger.info(f"   • Помилок (в цьому файлі): {stats.get('error_count', 0):,}")

    # Фінальна статистика
    final_count = 0
    try:
        final_count = system.vector_db.get_collection_size()
    except Exception as e:
        logger.error(f"Не вдалося отримати фінальний розмір колекції: {e}")

    processing_time_seconds = (datetime.now() - overall_start_time).total_seconds()

    logger.info("\n" + "="*60)
    logger.info("📊 ФІНАЛЬНА СТАТИСТИКА ЗАВАНТАЖЕННЯ:")
    logger.info("="*60)

    logger.info(f"✅ Всього записів у базі: {final_count:,}")
    # Records added depends on whether it was update or recreate, and initial_count
    if update_mode: # If we were in update mode for the whole process
        logger.info(f"📈 Додано нових записів (приблизно): {final_count - initial_count:,}")
    else: # If we were in recreate mode
        logger.info(f"📈 Всього записів після перестворення: {final_count:,}")

    logger.info(f"📦 Всього оброблено та спробувано індексувати: {total_indexed_overall:,}")
    logger.info(f"⏭️ Всього пропущено (можливо, дублікатів): {total_skipped_overall:,}")
    logger.info(f"❌ Всього помилок під час обробки файлів: {total_errors_overall:,}")
    logger.info(f"⏱️ Загальний час обробки: {processing_time_seconds:.1f} сек")
    
    if total_indexed_overall > 0 and processing_time_seconds > 0:
        logger.info(f"🚀 Швидкість обробки: {total_indexed_overall / processing_time_seconds:.0f} записів/сек")
    
    if fast_mode: # Assuming fast_mode implies indexing might be deferred
        logger.warning(f"\n⚠️  УВАГА (ШВИДКИЙ РЕЖИМ):")
        logger.warning(f"   🔥 Індексація могла бути ВІДКЛАДЕНА або ВИМКНЕНА під час завантаження.")
        logger.warning(f"   📋 Для ефективного пошуку може знадобитися увімкнути/завершити індексацію.")
        logger.warning(f"   🛠️ Запустіть відповідний скрипт (наприклад, enable_indexing.py), якщо потрібно.")
    
    logger.info("="*60)
    
    return True


# Main execution block
if __name__ == "__main__":
    # Default files and parameters for direct script execution
    DEFAULT_FILES = [
        "data/out_10_nodup_nonull.jsonl", # Assuming a data directory
        "data/out_12_nodup_nonull.jsonl"
    ]
    DEFAULT_CATEGORIES_FILE = "data/categories.jsonl" # Assuming a data directory
    DEFAULT_COLLECTION_NAME = "tender_vectors"
    DEFAULT_BATCH_SIZE = 1850
    DEFAULT_QDRANT_HOST = "localhost"
    DEFAULT_QDRANT_PORT = 6333

    # Script behavior parameters
    # UPDATE_MODE_CHOICE: None (ask), True (update), False (recreate)
    # For direct execution, let's default to asking the user if the collection exists.
    USER_CHOICE_UPDATE_MODE = None
    FORCE_RECREATE_COLLECTION = False # Set to True to always delete and recreate
    ENABLE_FAST_MODE = True           # Use fast processing logic

    # Setup basic logging for standalone script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("🚀 Запуск скрипта створення/оновлення векторної бази даних...")

    # Example: Check if default files exist, use placeholders if not
    actual_files_to_process = []
    for f_path_str in DEFAULT_FILES:
        f_path = Path(f_path_str)
        if f_path.exists():
            actual_files_to_process.append(f_path_str)
        else:
            logger.warning(f"⚠️  Файл даних за замовчуванням не знайдено: {f_path_str}. Його буде пропущено.")

    if not Path(DEFAULT_CATEGORIES_FILE).exists():
        logger.warning(f"⚠️  Файл категорій за замовчуванням не знайдено: {DEFAULT_CATEGORIES_FILE}. Система може використати стандартний або не мати категорій.")

    if not actual_files_to_process:
        logger.error("❌ Немає файлів даних для обробки. Завершення роботи.")
    else:
        logger.info("="*50)
        if ENABLE_FAST_MODE:
            logger.info("⚡ Режим ШВИДКОГО ЗАВАНТАЖЕННЯ УВІМКНЕНО.")
            logger.info("   Це може означати відкладену індексацію та оптимізації для швидкості.")
        else:
            logger.info("🐌 Режим звичайного завантаження.")
        logger.info(f"🗂️  Файли для обробки: {actual_files_to_process}")
        logger.info(f"🏷️  Колекція: {DEFAULT_COLLECTION_NAME}")
        logger.info(f"📦 Розмір батчу: {DEFAULT_BATCH_SIZE}")
        logger.info(f"🗄️  Qdrant: {DEFAULT_QDRANT_HOST}:{DEFAULT_QDRANT_PORT}")
        logger.info("="*50)

        # Confirmation from user
        response = input(f"\n🚀 Почати процес для колекції '{DEFAULT_COLLECTION_NAME}'? (y/n): ")
        if response.lower() == 'y':
            success = create_optimized_vector_database(
                jsonl_files=actual_files_to_process,
                categories_file=DEFAULT_CATEGORIES_FILE,
                collection_name=DEFAULT_COLLECTION_NAME,
                batch_size=DEFAULT_BATCH_SIZE,
                update_mode=USER_CHOICE_UPDATE_MODE, # Will ask if None and collection exists
                force_recreate=FORCE_RECREATE_COLLECTION,
                fast_mode=ENABLE_FAST_MODE,
                qdrant_host=DEFAULT_QDRANT_HOST,
                qdrant_port=DEFAULT_QDRANT_PORT
                # monitor_interval and max_records are not used in the new signature, handled internally or by data size
            )

            if success:
                logger.info("\n" + "="*60)
                logger.info("🎉 ПРОЦЕС СТВОРЕННЯ/ОНОВЛЕННЯ ВЕКТОРНОЇ БАЗИ ЗАВЕРШЕНО!")
                logger.info("="*60)
                if ENABLE_FAST_MODE:
                    logger.warning("⚠️  УВАГА: Якщо було використано швидкий режим, індексація може бути не завершена.")
                    logger.warning("   Перевірте статус та, за потреби, запустіть enable_indexing.py.")
                logger.info("="*60)
            else:
                logger.error("\n❌ Процес створення/оновлення векторної бази завершено з помилками або був скасований.")
        else:
            logger.info("❌ Операція скасована користувачем.")
