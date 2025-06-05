#!/usr/bin/env python3
"""
Оптимізований скрипт для створення векторної бази
"""

import logging
import json
from pathlib import Path
from tender_analysis_system import TenderAnalysisSystem
from qdrant_client import QdrantClient
from tqdm import tqdm  # Додайте імпорт на початку файлу

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

def process_file_with_stats(file_path, system, batch_size=1000):
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
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Завантаження"):
            try:
                data.append(json.loads(line.strip()))
            except Exception as e:
                pass  # Можна рахувати помилки тут
    
    # Індексація
    stats = system.vector_db.index_tenders(
        data,
        update_mode=True,
        batch_size=batch_size
    )
    
    return stats

def create_optimized_vector_database(
    jsonl_files: list,
    categories_file: str = "categories.jsonl",
    collection_name: str = "tender_vectors",
    batch_size: int = 2000,  # Збільшено
    max_records: int = None,
    monitor_interval: int = 50000
):
    """
    Створення оптимізованої векторної бази
    """
    
    print("="*60)
    print("🚀 ОПТИМІЗОВАНА ВЕКТОРНА БАЗА ТЕНДЕРІВ")
    print("="*60)
    
    # 1. Видалення старої колекції (якщо є)
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.delete_collection(collection_name)
        print(f"🗑️ Видалено стару колекцію '{collection_name}'")
    except:
        print(f"ℹ️ Колекція '{collection_name}' не існувала")
    
    # 2. Ініціалізація системи
    print("\n📦 Ініціалізація системи...")
    
    # Відключаємо логування від transformers
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    system = TenderAnalysisSystem(
        categories_file=categories_file,
        qdrant_host="localhost",
        qdrant_port=6333
    )
    
    if not system.initialize_system():
        print("❌ Помилка ініціалізації!")
        return False
    
    # Встановлення нової колекції
    system.vector_db.collection_name = collection_name
    
    # 3. Обробка файлів
    total_indexed = 0
    total_skipped = 0
    total_errors = 0

    for idx, jsonl_file in enumerate(jsonl_files, 1):
        print(f"\n📁 Файл {idx}/{len(jsonl_files)}: {Path(jsonl_file).name}")
        if not Path(jsonl_file).exists():
            print(f"❌ Файл не знайдено: {jsonl_file}")
            continue

        # Використовуємо нову функцію
        stats = process_file_with_stats(jsonl_file, system, batch_size=batch_size)
        total_indexed += stats.get('indexed_count', 0)
        total_skipped += stats.get('skipped_count', 0)
        total_errors += stats.get('error_count', 0)

        print(f"✅ Файл оброблено. Проіндексовано: {stats.get('indexed_count', 0):,}")

    # 4. Оптимізація колекції
    print("\n🔧 Оптимізація колекції...")
    system.vector_db.optimize_collection()
    
    # 5. Фінальна статистика
    print("\n" + "="*60)
    print("📊 ФІНАЛЬНА СТАТИСТИКА:")
    
    final_stats = system.vector_db.get_collection_stats()
    print(f"✅ Всього записів: {final_stats['points_count']:,}")
    print(f"📦 Розмір векторів: {final_stats['vectors_size_gb']} ГБ")
    print(f"📦 Розмір метаданих: {final_stats['payload_size_gb']} ГБ")
    print(f"💾 Загальний розмір: {final_stats['estimated_total_gb']} ГБ")
    print(f"⏭️ Пропущено дублікатів: {total_skipped:,}")
    print(f"❌ Помилок: {total_errors:,}")
    
    # Перевірка ефективності
    if final_stats['points_count'] > 0:
        bytes_per_point = (final_stats['estimated_total_gb'] * 1024**3) / final_stats['points_count']
        print(f"📏 Байт на запис: {bytes_per_point:.0f}")
        
        # Попередження якщо розмір великий
        if bytes_per_point > 5000:
            print("⚠️ УВАГА: Розмір на запис завеликий! Перевірте оптимізацію.")
    
    print("="*60)
    
    return True


if __name__ == "__main__":
    import json
    
    # ===== НАЛАШТУВАННЯ =====
    
    # Ваші файли
    FILES = [
        "out_10_nodup.jsonl",
        "out_12_nodup.jsonl"
    ]
    
    # Параметри
    COLLECTION_NAME = "tender_vectors"  # Нова назва!
    BATCH_SIZE = 1500  # Збільшено для швидкості
    MONITOR_INTERVAL = 100000  # Моніторинг кожні 100к записів
    
    # Для тестування
    MAX_RECORDS = None  # None = всі записи, або число для тесту
    
    # ===== ЗАПУСК =====
    
    print("🔧 ОПТИМІЗОВАНА ВЕРСІЯ")
    print(f"\n📁 Файли для обробки:")
    for f in FILES:
        print(f"  - {f}")
    
    print(f"\n⚙️ Параметри:")
    print(f"  - Колекція: {COLLECTION_NAME}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Monitor interval: {MONITOR_INTERVAL:,}")
    print(f"  - Max records: {MAX_RECORDS or 'Всі'}")
    
    response = input("\nПродовжити? (y/n): ")
    if response.lower() == 'y':
        create_optimized_vector_database(
            jsonl_files=FILES,
            collection_name=COLLECTION_NAME,
            batch_size=BATCH_SIZE,
            max_records=MAX_RECORDS,
            monitor_interval=MONITOR_INTERVAL
        )
    else:
        print("❌ Операція скасована")