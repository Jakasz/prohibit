# create_profiles_with_clusters.py
import json
import logging
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pickle
from tqdm import tqdm
from supplier_profiler import SupplierProfiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileBuilderWithClusters:
    def __init__(self, vector_db, category_mappings_file='data/categories_map.json'):
        self.vector_db = vector_db
        self.use_cache = True
        self.category_mappings = self._load_category_mappings(category_mappings_file)
        self.cache_file = "all_data_cache.pkl"
        self.profiles = {}
        
    def _load_category_mappings(self, filepath):
        """Завантаження маппінгу категорій"""
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Файл {filepath} не знайдено. Кластери не будуть використовуватися.")
            return {}
    

    def _validate_cache(self, supplier_data):
        """Перевірка валідності кешу"""
        if not supplier_data:
            return False
        
        # Перевіряємо що є дані
        total_items = sum(len(items) for items in supplier_data.values())
        logger.info(f"📋 Кеш містить {len(supplier_data)} постачальників з {total_items} записами")
        
        # Можна додати додаткові перевірки (наприклад, дату створення кешу)
        cache_stat = os.stat(self.cache_file)
        cache_age_days = (os.path.getmtime(self.cache_file) - cache_stat.st_mtime) / 86400
        
        if cache_age_days > 7:  # Кеш старший 7 днів
            logger.warning(f"⚠️ Кеш застарілий ({cache_age_days:.1f} днів)")
            response = input("Використати застарілий кеш? (y/n): ")
            return response.lower() == 'y'
        
        return True

    def _build_profiles(self, supplier_data=None):
        if supplier_data is None:
            logger.info("Error: supplier_data не передано!")
            return
        
        # Створюємо профілі
        for edrpou, items in tqdm(supplier_data.items(), desc="Створення профілів"):
            supplier_data = SupplierProfiler()
            profile = supplier_data.create_profile(edrpou, items)
            if profile:
                self.profiles[edrpou] = profile
        
        # Додаємо кластери та конкурентів
        self._add_clusters_and_competitors()
        return self.profiles


    def _load_data_with_cache(self):
        """Завантаження даних з кешу або векторної бази"""
        supplier_data = defaultdict(list)
        
        # Перевіряємо наявність кешу
        if self.use_cache and os.path.exists(self.cache_file):
            logger.info("📦 Знайдено кеш даних, завантажуємо...")
            try:
                with open(self.cache_file, 'rb') as f:
                    supplier_data = pickle.load(f)
                logger.info(f"✅ Завантажено {len(supplier_data)} постачальників з кешу")
                
                # Перевіряємо валідність кешу
                if self._validate_cache(supplier_data):                    
                   self.profiles = self._build_profiles(supplier_data)
                   return self.profiles
                else:
                    logger.warning("⚠️ Кеш не валідний, завантажуємо з бази...")
            except Exception as e:
                logger.error(f"❌ Помилка завантаження кешу: {e}")
                logger.info("📊 Завантажуємо дані з векторної бази...")
        else:
            logger.info("📦 Кеш не знайдено, завантажуємо з бази...")

        # Завантажуємо з векторної бази
        logger.info("📊 Завантаження даних з векторної бази...")
        self.profiles = self.build_profiles_from_vector_db()
        return self.profiles
    
    def build_profiles_from_vector_db(self):
        """Створення профілів з векторної бази з нуля"""
        logger.info("🚀 Створення профілів постачальників з векторної бази...")
        
        # Збираємо дані по постачальниках
        supplier_data = defaultdict(list)        
        offset = None
        total_records = 0
         
        while True:
            try:
                records, next_offset = self.vector_db.client.scroll(
                    collection_name=self.vector_db.collection_name,
                    offset=offset,
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not records:
                    break
                
                for record in records:
                    if record.payload:
                        edrpou = record.payload.get('edrpou', '')
                        if edrpou:
                            supplier_data[edrpou].append(record.payload)
                            total_records += 1
                
                if not next_offset:
                    break
                offset = next_offset
                
                logger.info(f"Завантажено {total_records:,} записів...")
                
            except Exception as e:
                logger.error(f"Помилка завантаження: {e}")
                break
        
        logger.info(f"✅ Знайдено {len(supplier_data)} унікальних постачальників")
        
        self.profiles = self._build_profiles(supplier_data)
        return self.profiles
    
    
    def _get_clusters_for_supplier(self, profile):
        """Визначення кластерів для постачальника"""
        clusters = set()
        
        # Отримуємо категорії незалежно від типу
        categories = {}
        
        if isinstance(profile, dict):
            categories = profile.get('categories', {})
        elif hasattr(profile, 'categories'):
            categories = profile.categories if isinstance(profile.categories, dict) else {}
        else:
            self.logger.warning(f"Профіль не має категорій: {type(profile)}")
            return []
        
        # Шукаємо в маппінгу
        for category in categories.keys():
            for cluster_name, cluster_categories in self.category_mappings.items():
                if category in cluster_categories:
                    clusters.add(cluster_name)
        
        return list(clusters)


    
    def _add_clusters_and_competitors(self):
        """Додавання кластерів та пошук конкурентів"""
        logger.info("🔄 Додаємо кластери та шукаємо конкурентів...")
        
        # Крок 1: Додаємо кластери
        for edrpou, profile in self.profiles.items():
            clusters = self._get_clusters_for_supplier(profile)
            
            # Додаємо кластери в залежності від типу профілю
            if hasattr(profile, 'clusters'):  # SupplierProfile об'єкт
                profile.clusters = clusters
            elif isinstance(profile, dict):  # Словник
                profile['clusters'] = clusters

        # Крок 2: Знаходимо конкурентів
        from tqdm import tqdm
        for edrpou, profile in tqdm(self.profiles.items(), desc="Аналіз конкурентів"):
            # Отримуємо кластери
            if hasattr(profile, 'clusters'):
                supplier_clusters = profile.clusters
            else:
                supplier_clusters = profile.get('clusters', [])
            
            if not supplier_clusters:
                if hasattr(profile, 'top_competitors'):
                    profile.top_competitors = []
                    profile.bottom_competitors = []
                else:
                    profile['top_competitors'] = []
                    profile['bottom_competitors'] = []
                continue
            
            # Знаходимо конкурентів
            competitors = []
            
            for comp_edrpou, comp_profile in self.profiles.items():
                if comp_edrpou == edrpou:
                    continue
                
                # Отримуємо кластери конкурента
                if hasattr(comp_profile, 'clusters'):
                    comp_clusters = comp_profile.clusters
                else:
                    comp_clusters = comp_profile.get('clusters', [])
                
                common_clusters = set(supplier_clusters) & set(comp_clusters)
                
                if common_clusters:
                    # Отримуємо метрики
                    if hasattr(comp_profile, 'metrics'):  # SupplierProfile
                        metrics = comp_profile.metrics
                        competitor_data = {
                            'edrpou': comp_edrpou,
                            'name': comp_profile.name,
                            'year_wins': metrics.won_positions,
                            'year_total': metrics.total_positions,
                            'year_win_rate': metrics.position_win_rate,
                            'common_clusters': list(common_clusters)
                        }
                    else:  # Словник
                        metrics = comp_profile.get('metrics', {})
                        competitor_data = {
                            'edrpou': comp_edrpou,
                            'name': comp_profile.get('name', ''),
                            'year_wins': metrics.get('won_positions', 0),
                            'year_total': metrics.get('total_positions', 0),
                            'year_win_rate': metrics.get('position_win_rate', 0),
                            'common_clusters': list(common_clusters)
                        }
                    
                    competitors.append(competitor_data)
            
            # Сортуємо за кількістю перемог
            competitors.sort(key=lambda x: x['year_wins'], reverse=True)
            
            # Топ 10 та найгірші 10
            top_10 = competitors[:10]
            bottom_10 = [c for c in reversed(competitors) if c['year_total'] > 0][:10]
            
            # Зберігаємо результати
            if hasattr(profile, 'top_competitors'):
                profile.top_competitors = top_10
            if hasattr(profile, 'bottom_competitors'):
                profile.bottom_competitors = bottom_10

            

    
    def save_profiles(self, filepath='supplier_profiles_with_clusters.json'):
        """Збереження профілів"""
        # Конвертуємо всі профілі в словники перед збереженням
        profiles_to_save = {}
        
        for edrpou, profile in self.profiles.items():
            if hasattr(profile, 'to_dict'):
                # Це SupplierProfile об'єкт
                profiles_to_save[edrpou] = profile.to_dict()
            elif isinstance(profile, dict):
                # Це вже словник
                profiles_to_save[edrpou] = profile
            else:
                logger.warning(f"Невідомий тип профілю для {edrpou}: {type(profile)}")
                continue
        
        # Зберігаємо
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profiles_to_save, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Збережено {len(profiles_to_save)} профілів в {filepath}")


# Основна функція
def create_or_update_profiles(system):
    """Створення або оновлення профілів з кластерами"""
    
    # Перевіряємо наявність існуючих профілів
    existing_profiles = [
        'supplier_profiles_with_clusters.json',
        'supplier_profiles_COMPLETE.json'
    ]
    
    profiles_exist = any(Path(f).exists() for f in existing_profiles)
    
    if profiles_exist:
        logger.info("📂 Знайдено існуючі профілі")
        response = input("Оновити існуючі профілі (u) чи створити нові (n)? [u/n]: ")
        
        if response.lower() == 'u':
            # Запускаємо оновлення
            from update_supplier_profiles_with_clusters import update_profiles_with_clusters_and_competitors
            update_profiles_with_clusters_and_competitors()
            return
    
    # Створюємо нові профілі
    logger.info("🆕 Створення нових профілів з кластерами...")
    
    builder = ProfileBuilderWithClusters(system.vector_db)
    profiles = builder._load_data_with_cache()
    builder.save_profiles()
    
    # Статистика
    logger.info("\n📊 Статистика:")
    logger.info(f"Всього профілів: {len(profiles)}")
    
    cluster_count = defaultdict(int)
    for profile in profiles.values():
        for cluster in profile.clusters:
            cluster_count[cluster] += 1
    
    logger.info("\nТоп кластери:")
    for cluster, count in sorted(cluster_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {cluster}: {count} постачальників")

if __name__ == "__main__":
    # Ініціалізація системи
    from tender_analysis_system import TenderAnalysisSystem
    
    system = TenderAnalysisSystem(
        categories_file="categories.jsonl",
        qdrant_host="localhost",
        qdrant_port=6333
    )
    system.initialize_system()
    
    # Створення або оновлення профілів
    create_or_update_profiles(system)