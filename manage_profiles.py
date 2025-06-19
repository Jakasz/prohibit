# manage_profiles.py
#!/usr/bin/env python3
"""
Універсальний менеджер профілів постачальників
"""
import argparse
import logging
from pathlib import Path
from tender_analysis_system import TenderAnalysisSystem
from create_profiles_with_clusters import ProfileBuilderWithClusters
from update_supplier_profiles_with_clusters import update_profiles_with_clusters_and_competitors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Управління профілями постачальників')
    parser.add_argument('action', choices=['create', 'update', 'check'], 
                        help='Дія: create - створити нові, update - оновити існуючі, check - перевірити')
    parser.add_argument('--force', action='store_true', 
                        help='Примусово перестворити профілі')
    
    args = parser.parse_args()
    
    # Ініціалізація системи
    logger.info("🚀 Ініціалізація системи...")
    system = TenderAnalysisSystem(
        categories_file="categories.jsonl",
        qdrant_host="localhost",
        qdrant_port=6333
    )
    system.initialize_system()
    
    if args.action == 'check':
        # Перевірка наявності профілів
        files = [
            'supplier_profiles_with_clusters.json',
            'supplier_profiles_COMPLETE.json'
        ]
        
        for f in files:
            if Path(f).exists():
                import json
                with open(f, 'r') as file:
                    data = json.load(file)
                    logger.info(f"✅ {f}: {len(data)} профілів")
                    
                    # Перевірка кластерів
                    with_clusters = sum(1 for p in data.values() if 'clusters' in p)
                    logger.info(f"   З кластерами: {with_clusters}")
            else:
                logger.info(f"❌ {f}: не знайдено")
    
    elif args.action == 'create' or args.force:
        # Створення нових профілів
        builder = ProfileBuilderWithClusters(system.vector_db)
        profiles = builder.build_profiles_from_vector_db()
        builder.save_profiles()
        
    elif args.action == 'update':
        # Оновлення існуючих
        if Path('supplier_profiles_COMPLETE.json').exists():           
            update_profiles_with_clusters_and_competitors()
        else:
            logger.error("❌ Не знайдено файл для оновлення. Спочатку створіть профілі.")

if __name__ == "__main__":
    main()