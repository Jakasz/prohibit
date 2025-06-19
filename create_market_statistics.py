"""
Скрипт для створення ринкової статистики
"""

import logging
from pathlib import Path
from tender_analysis_system import TenderAnalysisSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Ініціалізація системи
    logger.info("🚀 Ініціалізація системи...")
    
    system = TenderAnalysisSystem(
        categories_file="categories.jsonl",
        qdrant_host="localhost",
        qdrant_port=6333
    )
    
    if not system.initialize_system():
        logger.error("❌ Помилка ініціалізації системи")
        return
    
    # Перевірка наявності даних
    db_size = system.vector_db.get_collection_size()
    logger.info(f"📊 Записів у векторній базі: {db_size:,}")
    
    if db_size < 1000:
        logger.warning("⚠️ Недостатньо даних для статистики")
        return
    
    # Генерація статистики
    logger.info("📈 Генерація ринкової статистики...")
    results = system.update_market_statistics()
    
    logger.info(f"✅ Статистика згенерована!")
    logger.info(f"   • Категорій оброблено: {results['categories_processed']}")
    
    # Приклад статистики для кількох категорій
    if system.market_stats.category_stats:
        logger.info("\n📊 Приклади статистики:")
        
        for i, (category, stats) in enumerate(list(system.market_stats.category_stats.items())[:5]):
            logger.info(f"\n{i+1}. {category}:")
            logger.info(f"   • Тендерів: {stats['total_tenders']}")
            logger.info(f"   • Постачальників: {stats['total_suppliers']}")
            logger.info(f"   • Середня конкуренція: {stats['avg_suppliers_per_tender']:.1f} учасників/тендер")
            logger.info(f"   • Win rate новачків: {stats['new_supplier_win_rate']:.1%}")
            logger.info(f"   • Відкритість ринку: {stats['market_openness']:.1%}")


if __name__ == "__main__":
    main()