# ЗУПИНІТЬ поточний процес (Ctrl+C) і запустіть це:

from asyncio.log import logger
from collections import defaultdict
import json
import os
from typing import Dict, List

from tqdm import tqdm
from supplier_profiler import SupplierProfile
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
        
    def emergency_load_and_build(self):
        """АВАРІЙНИЙ РЕЖИМ - завантаження і побудова одночасно"""
        logger.info("🚨 АВАРІЙНИЙ РЕЖИМ ПОБУДОВИ")
        
        # 1. Спробуємо завантажити кешований дамп якщо є
        cache_file = "all_data_cache.pkl"
        supplier_data = defaultdict(list)
        
        if os.path.exists(cache_file):
            logger.info("📦 Знайдено кеш даних, завантажуємо...")
            with open(cache_file, 'rb') as f:
                supplier_data = pickle.load(f)
            logger.info(f"✅ Завантажено {len(supplier_data)} постачальників з кешу")
        else:
            # 2. ПАРАЛЕЛЬНЕ завантаження
            logger.info("⚡ Початок ШВИДКОГО паралельного завантаження...")
            
            # Отримаємо загальну кількість
            collection_info = self.vector_db.client.get_collection(
                collection_name=self.vector_db.collection_name
            )
            total_points = collection_info.points_count
            logger.info(f"📊 Всього записів в БД: {total_points:,}")
            
            # Розбиваємо на чанки для паралельної обробки
            chunk_size = 5000
            num_chunks = (total_points // chunk_size) + 1
            
            pbar = tqdm(total=total_points, desc="Завантаження даних", unit="записів")
            
            # Функція для завантаження одного чанка
            def load_chunk(offset):
                try:
                    records, _ = self.vector_db.client.scroll(
                        collection_name=self.vector_db.collection_name,
                        offset=offset,
                        limit=chunk_size,
                        with_payload=True,
                        with_vectors=False
                    )
                    return records
                except Exception as e:
                    logger.error(f"Помилка завантаження чанка {offset}: {e}")
                    return []
            
            # Паралельне завантаження
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for i in range(num_chunks):
                    offset = i * chunk_size
                    future = executor.submit(load_chunk, offset)
                    futures.append(future)
                
                for future in futures:
                    records = future.result()
                    # Групуємо по ЄДРПОУ
                    for record in records:
                        if record.payload and 'edrpou' in record.payload:
                            edrpou = record.payload['edrpou']
                            supplier_data[edrpou].append(record.payload)
                            pbar.update(1)
            pbar.close()

        # Зберігаємо кеш даних
        logger.info("💾 Зберігаємо кеш даних...")
        with open(cache_file, 'wb') as f:
            pickle.dump(dict(supplier_data), f)

        # 4. Побудова індексу: tender_number -> {edrpou: [items]}
        tender_participants = defaultdict(lambda: defaultdict(list))
        for edrpou, items in supplier_data.items():
            for item in items:
                tender = item.get('tender_number', '')
                if tender:
                    tender_participants[tender][edrpou].append(item)

        # 5. Довершуємо створення профілів якщо щось залишилось
        logger.info("🏁 Довершення створення профілів...")
        final_pbar = tqdm(supplier_data.items(), desc="Фіналізація профілів")
        batch_for_save = {}
        profiles_created = 0
        for edrpou, items in final_pbar:
            if edrpou not in self.profiler.profiles and len(items) > 0:
                try:
                    profile = self._create_profile(edrpou, items, tender_participants)
                    self.profiler.profiles[edrpou] = profile
                    batch_for_save[edrpou] = profile
                    profiles_created += 1

                    # Зберігаємо кожні 1000 профілів
                    if len(batch_for_save) >= 1000:
                        self._save_batch(batch_for_save, profiles_created)
                        batch_for_save = {}

                except Exception as e:
                    logger.error(f"Помилка профілю {edrpou}: {e}")
            final_pbar.set_postfix({'profiles': profiles_created})

        # Зберігаємо залишок
        if batch_for_save:
            self._save_batch(batch_for_save, profiles_created)

        # 6. Фінальне збереження
        self.profiler.save_profiles("supplier_profiles_COMPLETE.json")
        logger.info(f"✅ ГОТОВО! Створено {len(self.profiler.profiles)} профілів")
        
        return len(self.profiler.profiles)
    
    def _create_profile(self, edrpou: str, items: List[Dict], tender_participants) -> SupplierProfile:
        """МАКСИМАЛЬНО ШВИДКИЙ профіль - тільки критичне + конкуренти"""
        profile = SupplierProfile(
            edrpou=edrpou,
            name=items[0].get('supplier_name', '') if items else ''
        )

        # Підрахунок унікальних тендерів та виграних тендерів
        tenders = set()
        won_tenders = set()
        total_positions = 0
        won_positions = 0

        # Для конкурентних метрик
        competitors_won = set()
        competitors_lost = set()

        for item in items:
            tender = item.get('tender_number', '')
            if tender:
                tenders.add(tender)
            total_positions += 1
            my_win = bool(item.get('won'))
            if my_win and tender:
                won_positions += 1
                won_tenders.add(tender)
            elif my_win:
                won_positions += 1

            # --- Конкуренти ---
            if tender:
                for competitor, competitor_items in tender_participants[tender].items():
                    if competitor == edrpou:
                        continue
                    competitor_won = any(ci.get('won') for ci in competitor_items)
                    if my_win and not competitor_won:
                        competitors_won.add(competitor)
                    elif not my_win and competitor_won:
                        competitors_lost.add(competitor)

        # Ті, кого і вигравав, і програвав
        competitors_draw = competitors_won & competitors_lost
        competitors_won = competitors_won - competitors_draw
        competitors_lost = competitors_lost - competitors_draw

        profile.metrics.total_positions = total_positions
        profile.metrics.won_positions = won_positions
        profile.metrics.total_tenders = len(tenders)
        profile.metrics.won_tenders = len(won_tenders)

        # Відсоток виграних позицій
        if total_positions > 0:
            profile.metrics.position_win_rate = won_positions / total_positions
        else:
            profile.metrics.position_win_rate = 0.0

        # Відсоток виграних тендерів
        if len(tenders) > 0:
            profile.metrics.win_rate = len(won_tenders) / len(tenders)
        else:
            profile.metrics.win_rate = 0.0

        # Швидка оцінка надійності
        profile.reliability_score = min(
            profile.metrics.win_rate * 0.5 +
            min(len(tenders) / 50, 1.0) * 0.5,
            1.0
        )

        # --- Додаємо конкурентні метрики ---
        profile.competitors_won = list(competitors_won)
        profile.competitors_lost = list(competitors_lost)
        profile.competitors_draw = list(competitors_draw)

        return profile
    
    def _save_batch(self, batch: Dict, total_created: int):
        """Швидке збереження батчу"""
        filename = f"profiles_batch_{total_created}.json"
        
        # Зберігаємо тільки ключові метрики
        simple_batch = {}
        for edrpou, profile in batch.items():
            simple_batch[edrpou] = {
                'name': profile.name,
                'total_tenders': profile.metrics.total_tenders,
                'won_tenders': profile.metrics.won_tenders,
                'win_rate': profile.metrics.win_rate,
                'reliability': profile.reliability_score
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(simple_batch, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Збережено {filename}")





if __name__ == "__main__":
    system = TenderAnalysisSystem(
        categories_file="categories.jsonl",
        qdrant_host="localhost", 
        qdrant_port=6333
    )
    system.initialize_system()
    
    # Ультра швидкий білдер
    builder = UltraFastProfileBuilder(system)
    
    # ПОЇХАЛИ!
    total = builder.emergency_load_and_build()