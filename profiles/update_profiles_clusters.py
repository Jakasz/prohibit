# update_supplier_profiles_with_clusters.py
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Завантажуємо дані"""
    profiles_path = 'supplier_profiles_COMPLETE.json'
    mappings_path = 'data/categories_map.json'
    
    if not Path(profiles_path).exists():
        raise FileNotFoundError(f"Файл {profiles_path} не знайдено!")
    
    if not Path(mappings_path).exists():
        raise FileNotFoundError(f"Файл {mappings_path} не знайдено!")
    
    with open(profiles_path, 'r', encoding='utf-8') as f:
        supplier_profiles = json.load(f)
    
    with open(mappings_path, 'r', encoding='utf-8') as f:
        category_mappings = json.load(f)
    
    return supplier_profiles, category_mappings

def get_clusters_for_supplier(supplier_data, category_mappings):
    """Отримує список кластерів для постачальника на основі його категорій"""
    clusters = set()
    categories = supplier_data.get('categories', {})
    
    # Для кожної категорії постачальника
    for category in categories.keys():
        # Шукаємо в якому кластері вона є
        for cluster_name, cluster_categories in category_mappings.items():
            if category in cluster_categories:
                clusters.add(cluster_name)
    
    return list(clusters)

def find_competitors_by_clusters(supplier_edrpou, supplier_clusters, all_profiles):
    """Знаходить конкурентів які працюють в тих же кластерах"""
    competitors = {}
    
    for edrpou, profile in all_profiles.items():
        if edrpou == supplier_edrpou:
            continue
            
        # Отримуємо кластери конкурента
        competitor_clusters = profile.get('clusters', [])
        
        # Перевіряємо перетин кластерів
        common_clusters = set(supplier_clusters) & set(competitor_clusters)
        
        if common_clusters:
            # Використовуємо дані з профілю
            metrics = profile.get('metrics', {})
            year_wins = metrics.get('won_positions', 0)
            year_total = metrics.get('total_positions', 0)
            
            competitors[edrpou] = {
                'name': profile.get('name', ''),
                'common_clusters': list(common_clusters),
                'year_wins': year_wins,
                'year_total': year_total,
                'year_win_rate': year_wins / year_total if year_total > 0 else 0,
                'total_wins': year_wins,
                'total_positions': year_total
            }
    
    return competitors

def update_profiles_with_clusters_and_competitors():
    """Основна функція оновлення профілів"""
    try:
        supplier_profiles, category_mappings = load_data()
        
        logger.info(f"📊 Завантажено {len(supplier_profiles)} профілів постачальників")
        logger.info(f"📦 Завантажено {len(category_mappings)} кластерів категорій")
        
        # Крок 1: Додаємо кластери до кожного постачальника
        logger.info("🔄 Додаємо кластери до профілів...")
        
        for edrpou, profile in supplier_profiles.items():
            clusters = get_clusters_for_supplier(profile, category_mappings)
            profile['clusters'] = clusters
        
        # Крок 2: Знаходимо конкурентів для кожного постачальника
        logger.info("🔍 Аналізуємо конкурентів...")
        
        processed = 0
        for edrpou, profile in supplier_profiles.items():
            if processed % 1000 == 0:
                logger.info(f"  Оброблено {processed}/{len(supplier_profiles)} постачальників...")
            
            supplier_clusters = profile.get('clusters', [])
            if not supplier_clusters:
                profile['top_competitors'] = []
                profile['bottom_competitors'] = []
                processed += 1
                continue
            
            # Знаходимо всіх конкурентів
            competitors = find_competitors_by_clusters(edrpou, supplier_clusters, supplier_profiles)
            
            # Сортуємо за кількістю виграшів
            sorted_competitors = sorted(
                competitors.items(),
                key=lambda x: x[1]['year_wins'],
                reverse=True
            )
            
            # Топ 10 найкращих
            top_10 = []
            for comp_edrpou, comp_data in sorted_competitors[:10]:
                top_10.append({
                    'edrpou': comp_edrpou,
                    'name': comp_data['name'],
                    'year_wins': comp_data['year_wins'],
                    'year_total': comp_data['year_total'],
                    'year_win_rate': round(comp_data['year_win_rate'], 3),
                    'common_clusters': comp_data['common_clusters']
                })
            
            # Найгірші 10 (з ненульовою участю)
            bottom_10 = []
            for comp_edrpou, comp_data in reversed(sorted_competitors):
                if comp_data['year_total'] > 0 and len(bottom_10) < 10:
                    bottom_10.append({
                        'edrpou': comp_edrpou,
                        'name': comp_data['name'],
                        'year_wins': comp_data['year_wins'],
                        'year_total': comp_data['year_total'],
                        'year_win_rate': round(comp_data['year_win_rate'], 3),
                        'common_clusters': comp_data['common_clusters']
                    })
            
            profile['top_competitors'] = top_10
            profile['bottom_competitors'] = bottom_10
            processed += 1
        
        # Зберігаємо оновлені профілі
        logger.info("💾 Зберігаємо оновлені профілі...")
        output_path = 'supplier_profiles_with_clusters.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(supplier_profiles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Готово! Профілі збережено в {output_path}")
        
        # Статистика
        clusters_count = defaultdict(int)
        for profile in supplier_profiles.values():
            for cluster in profile.get('clusters', []):
                clusters_count[cluster] += 1
        
        logger.info("\n📊 Статистика по кластерах:")
        for cluster, count in sorted(clusters_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {cluster}: {count} постачальників")
            
    except Exception as e:
        logger.error(f"❌ Помилка: {e}")
        raise

if __name__ == "__main__":
    update_profiles_with_clusters_and_competitors()