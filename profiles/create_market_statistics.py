"""
Скрипт для генерації ринкової статистики
"""

import logging
import pickle
from pathlib import Path
from market_statistics import MarketStatistics
from category_manager import CategoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Перевірка наявності кешу
    cache_file = "files/all_data_cache.pkl"
    
    if not Path(cache_file).exists():
        logger.error(f"❌ Файл {cache_file} не знайдено!")
        logger.info("💡 Спробуйте спочатку запустити тренування для створення кешу")
        return
    
    # Ініціалізація компонентів
    logger.info("🚀 Ініціалізація компонентів...")
    
    # Менеджер категорій
    category_manager = CategoryManager("categories.jsonl")
    if Path("categories_map.json").exists():
        category_manager.load_category_mappings("categories_map.json")
    
    # Ринкова статистика
    market_stats = MarketStatistics(category_manager=category_manager)
    
    # Генерація статистики з кешу
    logger.info(f"📈 Генерація ринкової статистики з {cache_file}...")
    results = market_stats.calculate_market_statistics_from_cache(cache_file)
    
    if results:
        logger.info(f"✅ Статистика згенерована!")
        logger.info(f"   • Категорій оброблено: {results['categories_processed']}")
        
        # Приклад статистики для кількох категорій
        if market_stats.category_stats:
            logger.info("\n📊 Приклади статистики:")
            
            for i, (category, stats) in enumerate(list(market_stats.category_stats.items())[:5]):
                logger.info(f"\n{i+1}. {category}:")
                logger.info(f"   • Тендерів: {stats['total_tenders']:,}")
                logger.info(f"   • Постачальників: {stats['total_suppliers']:,}")
                logger.info(f"   • Середня конкуренція: {stats['avg_suppliers_per_tender']:.1f} учасників/тендер")
                logger.info(f"   • Win rate новачків: {stats['new_supplier_win_rate']:.1%}")
                logger.info(f"   • Відкритість ринку: {stats['market_openness']:.1%}")
                logger.info(f"   • Бар'єр входу: {stats['entry_barrier_score']:.1%}")
    else:
        logger.error("❌ Не вдалося згенерувати статистику")


if __name__ == "__main__":
    main()