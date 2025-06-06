
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

def create_optimized_vector_database(
    jsonl_files: list,
    categories_file: str = "categories.jsonl",
    collection_name: str = "tender_vectors",
    batch_size: int = 2000,
    max_records: int = None,
    monitor_interval: int = 50000,
    update_mode: bool = None,  # None = запитати користувача
    force_recreate: bool = False  # Примусове перестворення
):
    """
    Створення або оновлення оптимізованої векторної бази
    
    Args:
        jsonl_files: список JSONL файлів для обробки
        categories_file: файл з категоріями
        collection_name: назва колекції в Qdrant
        batch_size: розмір батчу для індексації
        max_records: максимальна кількість записів (для тестування)
        monitor_interval: інтервал моніторингу прогресу
        update_mode: True - оновлення, False - перестворення, None - запитати
        force_recreate: примусове перестворення без запиту
    """
    
    print("="*60)
    print("🚀 ОПТИМІЗОВАНА ВЕКТОРНА БАЗА ТЕНДЕРІВ")
    print("="*60)
    
    # 1. Перевірка існування колекції
    exists, collection_info = check_existing_collection(
        collection_name=collection_name
    )
    
    if exists:
        print(f"\n✅ Колекція '{collection_name}' вже існує:")
        print(f"   • Записів: {collection_info['points_count']:,}")
        print(f"   • Статус: {collection_info['status']}")
        print(f"   • Сегментів: {collection_info['segments_count']}")
        
        # Визначення режиму роботи
        if force_recreate:
            update_mode = False
            print("\n⚠️ УВАГА: Примусове перестворення колекції!")
        elif update_mode is None:
            print("\n🤔 Що робити з існуючою колекцією?")
            print("1. Оновити (додати нові записи)")
            print("2. Видалити і створити заново")
            print("3. Скасувати операцію")
            
            choice = input("\nВаш вибір (1/2/3): ")
            
            if choice == '1':
                update_mode = True
                print("✅ Режим оновлення")
            elif choice == '2':
                update_mode = False
                print("⚠️ Режим повного перестворення")
            else:
                print("❌ Операція скасована")
                return False
        
        # Видалення колекції якщо потрібно
        if not update_mode:
            try:
                client = QdrantClient(host="localhost", port=6333)
                client.delete_collection(collection_name)
                print(f"🗑️ Видалено стару колекцію '{collection_name}'")
            except Exception as e:
                print(f"❌ Помилка видалення колекції: {e}")
                return False
    else:
        print(f"\nℹ️ Колекція '{collection_name}' не існує, буде створена нова")
        update_mode = False
    
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
    
    # Встановлення колекції
    system.vector_db.collection_name = collection_name
    
    # 3. Початкова статистика
    initial_count = system.vector_db.get_collection_size() if update_mode else 0
    print(f"\n📊 Початкова кількість записів: {initial_count:,}")
    
    # 4. Обробка файлів
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

        # Обробка файлу з урахуванням режиму
        stats = process_file_with_stats(
            jsonl_file, 
            system, 
            batch_size=batch_size,
            update_mode=update_mode
        )
        
        total_indexed += stats.get('indexed_count', 0)
        total_skipped += stats.get('skipped_count', 0)
        total_errors += stats.get('error_count', 0)

        print(f"\n✅ Файл оброблено:")
        print(f"   • Проіндексовано: {stats.get('indexed_count', 0):,}")
        print(f"   • Пропущено дублікатів: {stats.get('skipped_count', 0):,}")
        print(f"   • Помилок: {stats.get('error_count', 0):,}")

    # 5. Оптимізація колекції
    if total_indexed > 0:
        print("\n🔧 Оптимізація колекції...")
        system.vector_db.optimize_collection()
    
    # 6. Фінальна статистика
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
    print(f"🚀 Швидкість: {(total_indexed + total_skipped) / processing_time:.0f} записів/сек")
    
    # Детальна статистика колекції
    final_stats = system.vector_db.get_collection_stats()
    if 'vectors_size_gb' in final_stats:
        print(f"\n💾 Розмір бази даних:")
        print(f"   • Вектори: {final_stats['vectors_size_gb']} ГБ")
        print(f"   • Метадані: {final_stats['payload_size_gb']} ГБ")
        print(f"   • Загальний розмір: {final_stats['estimated_total_gb']} ГБ")
        
        if final_stats['points_count'] > 0:
            bytes_per_point = (final_stats['estimated_total_gb'] * 1024**3) / final_stats['points_count']
            print(f"   • Байт на запис: {bytes_per_point:.0f}")
    
    print("="*60)
    
    return True


if __name__ == "__main__":
    # ===== НАЛАШТУВАННЯ =====
    
    # Ваші файли
    FILES = [
        "out_10_nodup.jsonl",
        "out_12_nodup.jsonl"
    ]
    
    # Параметри
    COLLECTION_NAME = "tender_vectors"  # Назва колекції
    BATCH_SIZE = 1500                   # Розмір батчу
    MONITOR_INTERVAL = 100000           # Інтервал моніторингу
    
    # Режими роботи (розкоментуйте потрібний)
    UPDATE_MODE = None      # Запитати користувача
    # UPDATE_MODE = True    # Завжди оновлювати
    # UPDATE_MODE = False   # Завжди перестворювати
    
    # Для тестування
    MAX_RECORDS = None  # None = всі записи, або число для тесту
    
    # ===== ІНФОРМАЦІЯ =====
    
    print("🔧 СИСТЕМА ВЕКТОРНОЇ БАЗИ ТЕНДЕРІВ")
    print("📅 Версія: 2.0 (з підтримкою оновлення)")
    
    print(f"\n📁 Файли для обробки:")
    for f in FILES:
        if Path(f).exists():
            size = Path(f).stat().st_size / (1024**3)
            print(f"  ✅ {f} ({size:.2f} ГБ)")
        else:
            print(f"  ❌ {f} (не знайдено)")
    
    print(f"\n⚙️ Параметри:")
    print(f"  - Колекція: {COLLECTION_NAME}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Monitor interval: {MONITOR_INTERVAL:,}")
    print(f"  - Max records: {MAX_RECORDS or 'Всі'}")
    
    if UPDATE_MODE is None:
        print(f"  - Режим: буде визначено автоматично")
    elif UPDATE_MODE:
        print(f"  - Режим: оновлення (додавання нових)")
    else:
        print(f"  - Режим: повне перестворення")
    
    # ===== ЗАПУСК =====
    
    response = input("\n🚀 Почати обробку? (y/n): ")
    if response.lower() == 'y':
        success = create_optimized_vector_database(
            jsonl_files=FILES,
            collection_name=COLLECTION_NAME,
            batch_size=BATCH_SIZE,
            max_records=MAX_RECORDS,
            monitor_interval=MONITOR_INTERVAL,
            update_mode=UPDATE_MODE
        )
        
        if success:
            print("\n✅ Операція завершена успішно!")
        else:
            print("\n❌ Операція завершена з помилками")
    else:
        print("❌ Операція скасована")