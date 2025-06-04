#!/usr/bin/env python3
"""
Безпечний скрипт для створення/оновлення векторної бази
"""

import logging
from pathlib import Path
from tender_analysis_system import TenderAnalysisSystem
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_collection_exists(host="localhost", port=6333, collection_name="tender_vectors"):
    """Перевірка існування колекції"""
    try:
        client = QdrantClient(host=host, port=port)
        collections = client.get_collections()
        return any(col.name == collection_name for col in collections.collections)
    except:
        return False

def create_or_update_vector_database(
    jsonl_files: list,
    categories_file: str = "categories.jsonl",
    collection_name: str = "tender_vectors",
    batch_size: int = 1000,
    recreate: bool = False,  # True = видалити стару базу
    max_records: int = None
):
    """
    Створення або оновлення векторної бази
    
    Args:
        recreate: True - видалити існуючу базу і створити нову
                 False - додати нові записи до існуючої
    """
    
    print("="*60)
    print("🚀 ВЕКТОРНА БАЗА ТЕНДЕРІВ")
    print("="*60)
    
    # 1. Перевірка існування колекції
    collection_exists = check_collection_exists(collection_name=collection_name)
    
    if collection_exists:
        print(f"ℹ️ Колекція '{collection_name}' вже існує")
        
        if recreate:
            print("⚠️ Режим RECREATE: існуюча колекція буде видалена")
            response = input("Продовжити? (y/n): ")
            if response.lower() != 'y':
                print("❌ Операція скасована")
                return False
            
            # Видалення колекції
            client = QdrantClient(host="localhost", port=6333)
            client.delete_collection(collection_name)
            print("🗑️ Колекція видалена")
            collection_exists = False
        else:
            print("📝 Режим UPDATE: нові записи будуть додані до існуючої бази")
            
            # Отримання поточної статистики
            client = QdrantClient(host="localhost", port=6333)
            info = client.get_collection(collection_name)
            print(f"📊 Поточний розмір бази: {info.points_count:,} записів")
    else:
        print(f"🆕 Колекція '{collection_name}' буде створена")
    
    # 2. Ініціалізація системи
    print("\n📦 Ініціалізація системи...")
    system = TenderAnalysisSystem(
        categories_file=categories_file,
        qdrant_host="localhost",
        qdrant_port=6333
    )
    
    if not system.initialize_system():
        print("❌ Помилка ініціалізації!")
        return False
    
    # 3. Встановлення правильної колекції
    if collection_exists and not recreate:
        # Використовуємо існуючу колекцію
        system.vector_db.collection_name = collection_name
    else:
        # Створюємо нову колекцію
        system.vector_db = system.vector_db.__class__(
            embedding_model=system.embedding_model,
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name=collection_name
        )
    
    # 4. Обробка файлів
    total_indexed = 0
    total_skipped = 0
    
    for idx, jsonl_file in enumerate(jsonl_files, 1):
        print(f"\n📁 Файл {idx}/{len(jsonl_files)}: {Path(jsonl_file).name}")
        
        if not Path(jsonl_file).exists():
            print(f"❌ Файл не знайдено: {jsonl_file}")
            continue
        
        file_size_mb = Path(jsonl_file).stat().st_size / (1024**2)
        print(f"📊 Розмір: {file_size_mb:.1f} MB")
        
        # Завантаження даних
        print("📥 Завантаження даних...")
        data = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_records and line_num > max_records:
                    print(f"⏹️ Досягнуто ліміт {max_records} записів")
                    break
                    
                try:
                    record = json.loads(line.strip())
                    data.append(record)
                    
                    if len(data) % 10000 == 0:
                        print(f"  Завантажено {len(data):,} записів...")
                        
                except Exception as e:
                    print(f"⚠️ Помилка в рядку {line_num}: {e}")
        
        print(f"✅ Завантажено {len(data):,} записів")
        
        # Індексація
        print("🔄 Індексація у векторну базу...")
        stats = system.vector_db.index_tenders(
            historical_data=data,
            update_mode=not recreate,  # update_mode=True якщо не recreate
            batch_size=batch_size
        )
        
        indexed = stats.get('indexed_count', 0)
        skipped = stats.get('skipped_count', 0)
        errors = stats.get('error_count', 0)
        
        total_indexed += indexed
        total_skipped += skipped
        
        print(f"✅ Проіндексовано: {indexed:,}")
        print(f"⏭️ Пропущено (дублікати): {skipped:,}")
        print(f"❌ Помилок: {errors:,}")
    
    # 5. Фінальна статистика
    print("\n" + "="*60)
    print("📊 ПІДСУМКИ:")
    print(f"✅ Нових записів додано: {total_indexed:,}")
    print(f"⏭️ Пропущено дублікатів: {total_skipped:,}")
    
    # Перевірка фінального розміру
    try:
        db_stats = system.vector_db.get_collection_stats()
        print(f"🗄️ Загальний розмір бази: {db_stats['points_count']:,} записів")
    except:
        pass
    
    print("="*60)
    
    return True


if __name__ == "__main__":
    import json
    
    # ===== НАЛАШТУВАННЯ =====
    
    # Ваші файли
    FILES = [
        "out_10.jsonl",
        "out_11.jsonl"
        # "path/to/your/second_file.jsonl"
    ]
    
    # Опції
    COLLECTION_NAME = "tender_vectors"  # Назва колекції
    BATCH_SIZE = 1700
    
    # ВАЖЛИВО: Режим роботи
    RECREATE_DATABASE = True  # False = додати до існуючої, True = створити нову
    
    # Для тестування
    MAX_RECORDS = None  # None = всі записи, число = обмеження
    
    # ===== ЗАПУСК =====
    
    print("🔧 Режим роботи:")
    if RECREATE_DATABASE:
        print("  ⚠️ RECREATE - існуюча база буде ВИДАЛЕНА і створена нова")
    else:
        print("  ✅ UPDATE - нові записи будуть ДОДАНІ до існуючої бази")
    
    print(f"\n📁 Файли для обробки:")
    for f in FILES:
        print(f"  - {f}")
    
    print(f"\n⚙️ Параметри:")
    print(f"  - Колекція: {COLLECTION_NAME}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max records: {MAX_RECORDS or 'Всі'}")
    
    response = input("\nПродовжити? (y/n): ")
    if response.lower() == 'y':
        create_or_update_vector_database(
            jsonl_files=FILES,
            collection_name=COLLECTION_NAME,
            batch_size=BATCH_SIZE,
            recreate=RECREATE_DATABASE,
            max_records=MAX_RECORDS
        )
    else:
        print("❌ Операція скасована")