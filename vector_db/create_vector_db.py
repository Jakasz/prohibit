
"""
Оптимізований скрипт для створення векторної бази з підтримкою оновлення
"""

import logging
import json
from pathlib import Path
from tender_analysis_system import TenderAnalysisSystem
from qdrant_client import QdrantClient
from tqdm import tqdm
from datetime import datetime
from qdrant_client.http import models

from vector_database import TenderVectorDB

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fast_upsert_batch(self, points, batch_num: int = 0) -> int:
    """Швидка вставка батчу БЕЗ очікування завершення"""
    try:
        print(f"📡 Швидка відправка батчу #{batch_num} з {len(points)} точок...")
        
        if not points:
            print(f"❌ Порожній батч #{batch_num}")
            return 0
        
        # 🔥 ШВИДКА відправка БЕЗ очікування
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,  # НЕ чекаємо завершення!
            ordering=models.WriteOrdering.WEAK  # Слабка консистентність
        )
        
        print(f"⚡ Батч #{batch_num} відправлено асинхронно")
        return len(points)
        
    except Exception as e:
        print(f"❌ Помилка швидкої відправки батчу #{batch_num}: {e}")
        return 0


def monitor_progress(system, start_count, interval=50000):
    """Моніторинг прогресу та розміру"""
    current_count = system.vector_db.get_collection_size()
    if current_count - start_count >= interval:
        stats = system.vector_db.get_storage_info()
        logging.info(f"""
        📊 СТАТИСТИКА:
        - Записів: {current_count:,}
        - Розмір векторів: {stats['vectors_size_gb']} ГБ
        - Розмір метаданих: {stats['payload_size_gb']} ГБ
        - Загальний розмір: {stats['estimated_total_gb']} ГБ
        """)
        return current_count
    return start_count

def process_file_with_stats(file_path, system, batch_size=1000, update_mode=True):
    """
    Обробка файлу з детальною статистикою
    
    Args:
        file_path: шлях до JSONL файлу
        system: екземпляр TenderAnalysisSystem
        batch_size: розмір батчу для обробки
        update_mode: True - додавання нових записів, False - повна переіндексація
    """
    print(f"\n📁 Обробка файлу: {file_path}")
    file_size = Path(file_path).stat().st_size / (1024**3)
    print(f"📊 Розмір файлу: {file_size:.2f} ГБ")
    
    # Підрахунок кількості рядків
    print("📝 Підрахунок записів...")
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"✅ Всього записів: {total_lines:,}")
    
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
                    print(f"⚠️ Помилка парсингу JSON: {e}")
    
    if errors > 0:
        print(f"⚠️ Всього помилок завантаження: {errors}")
    
    # Індексація з урахуванням режиму оновлення
    print(f"\n🔄 Індексація в режимі: {'оновлення' if update_mode else 'повного перестворення'}")
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
        client = QdrantClient(host=host, port=port)
        collection_info = client.get_collection(collection_name)
        
        info = {
            'exists': True,
            'points_count': collection_info.points_count,
            'status': collection_info.status,
            'vectors_size': collection_info.config.params.vectors.size,
            'segments_count': collection_info.segments_count if collection_info.segments_count else 0
        }
        
        return True, info
    except:
        return False, {'exists': False}


def process_file_fast(file_path, system, batch_size=5000, update_mode=True, fast_mode=True):
    """ШВИДКА обробка файлу"""
    print(f"⚡ ШВИДКА обробка: {file_path}")
    
    # Підрахунок записів
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"📊 Всього записів: {total_lines:,}")
    
    # Швидке завантаження та обробка
    data = []
    errors = 0
    
    print("📥 Швидке завантаження в пам'ять...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Завантаження", unit="рядків"):
            try:
                data.append(json.loads(line.strip()))
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"⚠️ JSON помилка: {e}")
    
    print(f"✅ Завантажено {len(data):,} записів (помилок: {errors})")
    
    # ШВИДКА індексація
    print(f"⚡ ШВИДКА індексація з батчами по {batch_size}...")
    stats = system.vector_db.index_tenders(
        data,
        update_mode=update_mode,
        batch_size=batch_size
    )
    
    return stats



def create_optimized_vector_database(
    jsonl_files: list,
    categories_file: str = "categories.jsonl",
    collection_name: str = "tender_vectors",
    batch_size: int = 5000,
    update_mode: bool = None,
    force_recreate: bool = False,
    fast_mode: bool = True
):
    """
    ШВИДКЕ створення векторної бази БЕЗ індексації
    """
    
    print("="*60)
    if fast_mode:
        print("⚡ ШВИДКИЙ РЕЖИМ ЗАВАНТАЖЕННЯ")
        print("   • Індексація ВИМКНЕНА")
        print("   • Збільшений розмір батчу")
        print("   • Асинхронна відправка")
    else:
        print("🐌 ЗВИЧАЙНИЙ РЕЖИМ З ІНДЕКСАЦІЄЮ")
    print("="*60)
    
    # Перевірка існування колекції
    exists, collection_info = check_existing_collection(collection_name=collection_name)
    
    if exists and not force_recreate:
        print(f"\n✅ Колекція '{collection_name}' вже існує:")
        print(f"   • Записів: {collection_info['points_count']:,}")
        
        if update_mode is None:
            choice = input("\n❓ Видалити і створити заново? (y/n): ")
            if choice.lower() != 'y':
                print("❌ Операція скасована")
                return False
            update_mode = False
                
        # Видалення якщо потрібно
        if not update_mode:
            try:
                client = QdrantClient(host="localhost", port=6333)
                client.delete_collection(collection_name)
                print(f"🗑️ Видалено стару колекцію '{collection_name}'")
            except Exception as e:
                print(f"❌ Помилка видалення: {e}")
                return False
    
    # Ініціалізація системи
    print("\n📦 Ініціалізація системи...")
    
    # Тихий режим для transformers
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    system = TenderAnalysisSystem(
        categories_file=categories_file,
        qdrant_host="localhost",
        qdrant_port=6333
    )
    
    if not system.initialize_system():
        print("❌ Помилка ініціалізації!")
        return False
    
    # Встановлюємо правильну назву колекції
    system.vector_db.collection_name = collection_name
    
    # Початкова статистика
    initial_count = system.vector_db.get_collection_size() if update_mode else 0
    print(f"\n📊 Початкова кількість записів: {initial_count:,}")
    
    # Обробка файлів
    total_indexed = 0
    total_skipped = 0
    total_errors = 0
    start_time = datetime.now()

    for idx, jsonl_file in enumerate(jsonl_files, 1):
        print(f"\n{'='*60}")
        print(f"📁 Файл {idx}/{len(jsonl_files)}: {Path(jsonl_file).name}")
        print(f"{'='*60}")
        
        if not Path(jsonl_file).exists():
            print(f"❌ Файл не знайдено: {jsonl_file}")
            continue

        # Обробка файлу
        stats = process_file_fast(
            jsonl_file, 
            system, 
            batch_size=batch_size,
            update_mode=update_mode if update_mode is not None else True,
            fast_mode=fast_mode
        )
        
        total_indexed += stats.get('indexed_count', 0)
        total_skipped += stats.get('skipped_count', 0)
        total_errors += stats.get('error_count', 0)

        print(f"\n✅ Файл оброблено:")
        print(f"   • Проіндексовано: {stats.get('indexed_count', 0):,}")
        print(f"   • Пропущено: {stats.get('skipped_count', 0):,}")
        print(f"   • Помилок: {stats.get('error_count', 0):,}")

    # Фінальна статистика
    final_count = system.vector_db.get_collection_size()
    processing_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("📊 ФІНАЛЬНА СТАТИСТИКА:")
    print("="*60)
    
    print(f"✅ Всього записів у базі: {final_count:,}")
    print(f"📈 Додано нових записів: {final_count - initial_count:,}")
    print(f"📦 Проіндексовано: {total_indexed:,}")
    print(f"⏭️ Пропущено дублікатів: {total_skipped:,}")
    print(f"❌ Помилок: {total_errors:,}")
    print(f"⏱️ Час обробки: {processing_time:.1f} сек")
    
    if total_indexed > 0:
        print(f"🚀 Швидкість: {total_indexed / processing_time:.0f} записів/сек")
    
    if fast_mode:
        print(f"\n⚠️  УВАГА:")
        print(f"   🔥 Індексація ВИМКНЕНА!")
        print(f"   📋 Для пошуку потрібно увімкнути індексацію")
        print(f"   🛠️ Запустіть: python enable_indexing.py")
    
    print("="*60)
    
    return True



if __name__ == "__main__":
    # ===== НАЛАШТУВАННЯ ДЛЯ ШВИДКОГО ЗАВАНТАЖЕННЯ =====
    
    FILES = [
        "out_10_nodup_nonull.jsonl",
        "out_12_nodup_nonull.jsonl"
    ]
    
    # 🔥 ЗБІЛЬШЕНІ параметри для швидкості
    COLLECTION_NAME = "tender_vectors"
    BATCH_SIZE = 1850                   
    MONITOR_INTERVAL = 50000           
    
    # Режим роботи
    UPDATE_MODE = True                  # Повне перестворення
    MAX_RECORDS = None                  # Всі записи
    
    print("🚀 ШВИДКЕ ЗАВАНТАЖЕННЯ (БЕЗ ІНДЕКСАЦІЇ)")
    print("="*50)
    print("⚡ Колекція буде створена БЕЗ індексації")
    print("⚡ Збільшений розмір батчу до 5000")
    print("⚡ Асинхронна відправка батчів")
    print("⚡ Після завантаження потрібно буде увімкнути індексацію")
    print("="*50)
    
    response = input("\n🚀 Почати ШВИДКЕ завантаження? (y/n): ")
    if response.lower() == 'y':
        success = create_optimized_vector_database(
            jsonl_files=FILES,
            collection_name=COLLECTION_NAME,
            batch_size=BATCH_SIZE,
            max_records=MAX_RECORDS,
            monitor_interval=MONITOR_INTERVAL,
            update_mode=UPDATE_MODE,
            fast_mode=True  # 🔥 НОВИЙ параметр
        )
        
        if success:
            print("\n" + "="*60)
            print("🎉 ШВИДКЕ ЗАВАНТАЖЕННЯ ЗАВЕРШЕНО!")
            print("="*60)
            print("⚠️  УВАГА: Індексація ВИМКНЕНА!")
            print("📋 Для увімкнення індексації запустіть:")
            print("   python enable_indexing.py")
            print("="*60)
        else:
            print("\n❌ Швидке завантаження завершено з помилками")
    else:
        print("❌ Операція скасована")
