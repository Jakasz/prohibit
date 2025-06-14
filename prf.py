# ЗУПИНІТЬ поточний процес (Ctrl+C) і запустіть це:

from asyncio.log import logger
from collections import defaultdict
import json
import logging
import os
from typing import Dict, List

from tqdm import tqdm
from supplier_profiler import SupplierMetrics, SupplierProfile
from tender_analysis_system import TenderAnalysisSystem


import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import pickle

class UltraFastProfileBuilder:    
    def __init__(self, system: TenderAnalysisSystem):
        self.system = system
        self.vector_db = system.vector_db
        self.profiler = system.supplier_profiler
        self.logger = logging.getLogger(__name__)


    def _save_batch(self, batch: Dict, total_created: int):
        """Збереження ПОВНОГО батчу профілів"""
        filename = f"profiles_batch_{total_created}.json"
        
        # Зберігаємо ПОВНІ профілі
        full_batch = {}
        for edrpou, profile in batch.items():
            full_batch[edrpou] = profile.to_dict()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(full_batch, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"💾 Збережено батч {filename} з {len(batch)} профілями")




    def emergency_load_and_build(self):
        """АВАРІЙНИЙ РЕЖИМ - завантаження і побудова одночасно"""
        self.logger.info("🚨 АВАРІЙНИЙ РЕЖИМ ПОБУДОВИ")
        
        # Перевірка profiler
        if self.profiler is None:
            self.logger.error("❌ profiler is None! Створюємо новий...")
            from supplier_profiler import SupplierProfiler
            self.profiler = SupplierProfiler()
        
        if not hasattr(self.profiler, 'profiles'):
            self.profiler.profiles = {}
        
        # 1. Спробуємо завантажити кешований дамп якщо є
        cache_file = "all_data_cache.pkl"
        supplier_data = defaultdict(list)
        
        if os.path.exists(cache_file):
            self.logger.info("📦 Знайдено кеш даних, завантажуємо...")
            with open(cache_file, 'rb') as f:
                supplier_data = pickle.load(f)
            self.logger.info(f"✅ Завантажено {len(supplier_data)} постачальників з кешу")
        else:
            # 2. ПРАВИЛЬНЕ завантаження через scroll
            self.logger.info("⚡ Початок правильного завантаження...")
            
            # Отримаємо загальну кількість
            collection_info = self.vector_db.client.get_collection(
                collection_name=self.vector_db.collection_name
            )
            total_points = collection_info.points_count
            self.logger.info(f"📊 Всього записів в БД: {total_points:,}")
            
            pbar = tqdm(total=total_points, desc="Завантаження даних", unit="записів")
            
            # ПРАВИЛЬНИЙ SCROLL
            offset = None
            total_loaded = 0
            batch_size = 10000  # Більший батч для швидкості
            
            while True:
                try:
                    # Використовуємо scroll правильно
                    records, next_offset = self.vector_db.client.scroll(
                        collection_name=self.vector_db.collection_name,
                        offset=offset,
                        limit=batch_size,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    if not records:
                        break
                    
                    # Групуємо по ЄДРПОУ
                    for record in records:
                        if record.payload:
                            # ВИПРАВЛЕННЯ: Перевіряємо різні варіанти полів
                            edrpou = (record.payload.get('edrpou') or 
                                    record.payload.get('EDRPOU') or 
                                    '')
                            
                            if edrpou:
                                # Зберігаємо весь payload
                                supplier_data[edrpou].append(record.payload)
                                total_loaded += 1
                                pbar.update(1)
                            else:
                                # Логуємо проблемні записи
                                self.logger.debug(f"Запис без ЄДРПОУ: {record.payload}")
                    
                    # Логування прогресу
                    if total_loaded % 50000 == 0:
                        self.logger.info(f"   Завантажено {total_loaded:,} записів, унікальних ЄДРПОУ: {len(supplier_data):,}")
                    
                    # Переходимо до наступного батчу
                    if not next_offset:
                        break
                    offset = next_offset
                    
                except Exception as e:
                    self.logger.error(f"Помилка завантаження: {e}")
                    break
            
            pbar.close()
            self.logger.info(f"✅ Завантажено {total_loaded:,} записів для {len(supplier_data):,} постачальників")

            # Зберігаємо кеш даних
            self.logger.info("💾 Зберігаємо кеш даних...")
            with open(cache_file, 'wb') as f:
                pickle.dump(dict(supplier_data), f)

        # 3. Діагностика даних
        self.logger.info("\n🔍 ДІАГНОСТИКА ДАНИХ:")
        # Перевіряємо перші кілька постачальників
        for i, (edrpou, items) in enumerate(list(supplier_data.items())[:5]):
            self.logger.info(f"ЄДРПОУ {edrpou}: {len(items)} записів")
            if items:
                first_item = items[0]
                self.logger.info(f"  Приклад полів: {list(first_item.keys())[:10]}")
                self.logger.info(f"  Назва: {first_item.get('supplier_name') or first_item.get('supp_name') or 'НЕ ЗНАЙДЕНО'}")

        # 4. Створення профілів
        self.logger.info("🏁 Створення профілів...")
        
        all_profiles = {}
        
        final_pbar = tqdm(supplier_data.items(), desc="Створення профілів")
        profiles_created = 0
        errors = 0
        
        for edrpou, items in final_pbar:
            if len(items) > 0:
                try:
                    # ВИКОРИСТОВУЄМО МЕТОД profiler.create_profile()
                    profile = self.profiler.create_profile(edrpou, items)
                    
                    if profile:
                        all_profiles[edrpou] = profile
                        self.profiler.profiles[edrpou] = profile
                        profiles_created += 1
                    else:
                        self.logger.warning(f"Профіль не створено для {edrpou}")
                        
                except Exception as e:
                    errors += 1
                    self.logger.error(f"Помилка профілю {edrpou}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
            final_pbar.set_postfix({
                'profiles': profiles_created,
                'errors': errors,
                'items': len(items)
            })
        
        # 5. ФІНАЛЬНЕ ЗБЕРЕЖЕННЯ ВСІХ ПРОФІЛІВ
        self.logger.info("💾 Збереження фінального файлу з усіма профілями...")
        
        # Використовуємо вбудований метод profiler для збереження
        self.profiler.save_profiles("supplier_profiles_COMPLETE.json")
        
        self.logger.info(f"✅ ГОТОВО! Створено {len(all_profiles)} профілів")
        self.logger.info(f"❌ Помилок: {errors}")
        self.logger.info(f"📁 Фінальний файл: supplier_profiles_COMPLETE.json")
        
        # 6. Діагностика результатів
        self.logger.info("\n📊 ДІАГНОСТИКА РЕЗУЛЬТАТІВ:")
        sorted_suppliers = sorted(supplier_data.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (edrpou, items) in enumerate(sorted_suppliers[:10]):
            profile = all_profiles.get(edrpou)
            if profile:
                self.logger.info(f"   #{i+1} ЄДРПОУ {edrpou}: {len(items)} записів в даних, "
                            f"{profile.metrics.total_positions} позицій в профілі, "
                            f"win_rate={profile.metrics.win_rate:.2%}")
        
        return len(all_profiles)
if __name__ == "__main__":
    system = TenderAnalysisSystem(
        categories_file="data/categories.jsonl",
        qdrant_host="localhost", 
        qdrant_port=6333
    )
    system.initialize_system()
    
    # Ультра швидкий білдер
    builder = UltraFastProfileBuilder(system)
    
    # ПОЇХАЛИ!
    total = builder.emergency_load_and_build()