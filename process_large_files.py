import json
import logging
from pathlib import Path
from datetime import datetime
import gc
import sys
from tender_analysis_system import TenderAnalysisSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_large_files(file_paths: list, 
                       categories_file: str = "categories.jsonl",
                       mapping_file: str = "category_mappings.json",
                       batch_size: int = 1000,
                       max_records_per_file: int = None):
    """
    Обробка великих JSONL файлів з оптимізацією пам'яті
    
    Args:
        file_paths: список шляхів до JSONL файлів
        categories_file: файл з категоріями
        mapping_file: файл з маппінгом категорій
        batch_size: розмір батчу для обробки
        max_records_per_file: максимум записів з кожного файлу (для тестування)
    """
    
    # 1. Ініціалізація системи
    logger.info("🚀 Ініціалізація системи...")
    system = TenderAnalysisSystem(categories_file=categories_file)
    
    if not system.initialize_system():
        logger.error("❌ Помилка ініціалізації системи")
        return
    
    # 2. Завантаження маппінгу категорій
    if Path(mapping_file).exists():
        logger.info(f"📂 Завантаження маппінгу категорій з {mapping_file}")
        system.categories_manager.load_category_mappings(mapping_file)
    
    # 3. Статистика
    total_stats = {
        'total_processed': 0,
        'total_indexed': 0,
        'total_errors': 0,
        'processing_time': 0,
        'file_stats': {}
    }
    
    overall_start = datetime.now()
    
    # 4. Обробка кожного файлу
    for file_idx, file_path in enumerate(file_paths):
        if not Path(file_path).exists():
            logger.error(f"❌ Файл {file_path} не знайдено")
            continue
        
        file_size_gb = Path(file_path).stat().st_size / (1024**3)
        logger.info(f"\n📁 Обробка файлу {file_idx + 1}/{len(file_paths)}: {file_path}")
        logger.info(f"📊 Розмір файлу: {file_size_gb:.2f} ГБ")
        
        file_start = datetime.now()
        file_stats = {
            'records_read': 0,
            'records_indexed': 0,
            'errors': 0,
            'batches': 0
        }
        
        # Батчева обробка файлу
        batch_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Обмеження для тестування
                    if max_records_per_file and line_num > max_records_per_file:
                        logger.info(f"⏹️ Досягнуто ліміт {max_records_per_file} записів")
                        break
                    
                    try:
                        # Парсинг JSON
                        record = json.loads(line.strip())
                        batch_data.append(record)
                        file_stats['records_read'] += 1
                        
                        # Обробка батчу
                        if len(batch_data) >= batch_size:
                            logger.info(f"🔄 Обробка батчу {file_stats['batches'] + 1} "
                                      f"({len(batch_data)} записів)...")
                            
                            # Індексація в векторну базу
                            index_stats = system.vector_db.index_tenders(
                                batch_data,
                                update_mode=True,  # Додаємо до існуючої бази
                                batch_size=100  # Менший батч для Qdrant
                            )
                            
                            file_stats['records_indexed'] += index_stats['indexed_count']
                            file_stats['errors'] += index_stats['error_count']
                            file_stats['batches'] += 1
                            
                            # Очищення пам'яті
                            batch_data.clear()
                            gc.collect()
                            
                            # Прогрес
                            if line_num % 10000 == 0:
                                elapsed = (datetime.now() - file_start).total_seconds()
                                speed = line_num / elapsed if elapsed > 0 else 0
                                logger.info(f"📈 Оброблено {line_num:,} рядків "
                                          f"({speed:.0f} рядків/сек)")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Помилка JSON в рядку {line_num}: {e}")
                        file_stats['errors'] += 1
                    except Exception as e:
                        logger.error(f"❌ Помилка обробки рядка {line_num}: {e}")
                        file_stats['errors'] += 1
                
                # Обробка останнього батчу
                if batch_data:
                    logger.info(f"🔄 Обробка останнього батчу ({len(batch_data)} записів)...")
                    index_stats = system.vector_db.index_tenders(
                        batch_data,
                        update_mode=True,
                        batch_size=100
                    )
                    file_stats['records_indexed'] += index_stats['indexed_count']
                    file_stats['errors'] += index_stats['error_count']
        
        except Exception as e:
            logger.error(f"❌ Критична помилка при обробці файлу: {e}")
        
        # Статистика файлу
        file_time = (datetime.now() - file_start).total_seconds()
        file_stats['processing_time'] = file_time
        
        logger.info(f"\n✅ Файл оброблено за {file_time:.1f} сек")
        logger.info(f"📊 Прочитано: {file_stats['records_read']:,}")
        logger.info(f"📊 Проіндексовано: {file_stats['records_indexed']:,}")
        logger.info(f"📊 Помилок: {file_stats['errors']:,}")
        
        # Оновлення загальної статистики
        total_stats['file_stats'][file_path] = file_stats
        total_stats['total_processed'] += file_stats['records_read']
        total_stats['total_indexed'] += file_stats['records_indexed']
        total_stats['total_errors'] += file_stats['errors']
        
        # Очищення пам'яті між файлами
        gc.collect()
    
    # 5. Фінальна статистика
    total_stats['processing_time'] = (datetime.now() - overall_start).total_seconds()
    
    logger.info("\n" + "="*50)
    logger.info("📊 ПІДСУМКОВА СТАТИСТИКА:")
    logger.info(f"⏱️ Загальний час: {total_stats['processing_time']:.1f} сек")
    logger.info(f"📄 Оброблено файлів: {len(file_paths)}")
    logger.info(f"📝 Всього записів: {total_stats['total_processed']:,}")
    logger.info(f"✅ Проіндексовано: {total_stats['total_indexed']:,}")
    logger.info(f"❌ Помилок: {total_stats['total_errors']:,}")
    
    # 6. Перевірка векторної бази
    db_stats = system.vector_db.get_collection_stats()
    logger.info(f"\n📊 Статистика векторної бази:")
    logger.info(f"🗄️ Всього записів: {db_stats['points_count']:,}")
    
    # 7. Збереження системи
    logger.info("\n💾 Збереження системи...")
    system.save_system("tender_system_large.pkl")
    
    return total_stats

# Використання
if __name__ == "__main__":
    # Ваші файли
    files = [
        "tender_data_part1.jsonl",  # 3.5 GB
        "tender_data_part2.jsonl"   # 3.5 GB
    ]
    
    # Спочатку створіть маппінг категорій
    from category_mapping import analyze_categories_and_create_mapping
    analyze_categories_and_create_mapping("categories.jsonl")
    
    # Потім запустіть обробку
    stats = process_large_files(
        file_paths=files,
        batch_size=1000,  # Можна збільшити якщо достатньо RAM
        max_records_per_file=None  # Для тесту можна поставити 10000
    )