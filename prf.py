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



    # def _save_batch(self, batch: Dict, total_created: int):
    #     """Збереження ПОВНОГО батчу профілів"""
    #     filename = f"profiles_batch_{total_created}.json"
        
    #     # Зберігаємо ПОВНІ профілі, а не спрощені
    #     full_batch = {}
    #     for edrpou, profile in batch.items():
    #         full_batch[edrpou] = profile.to_dict()  # Конвертуємо в повний словник
        
    #     with open(filename, 'w', encoding='utf-8') as f:
    #         json.dump(full_batch, f, ensure_ascii=False, indent=2)
        
    #     self.logger.info(f"💾 Збережено батч {filename} з {len(batch)} профілями")
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
                        if record.payload and 'edrpou' in record.payload:
                            edrpou = record.payload['edrpou']
                            supplier_data[edrpou].append(record.payload)
                            total_loaded += 1
                            pbar.update(1)
                    
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

        # 3. Побудова індексу: tender_number -> {edrpou: [items]}
        self.logger.info("🔨 Побудова індексу учасників тендерів...")
        tender_participants = defaultdict(lambda: defaultdict(list))
        for edrpou, items in supplier_data.items():
            for item in items:
                tender = item.get('tender_number', '')
                if tender:
                    tender_participants[tender][edrpou].append(item)

        # 4. Створення профілів
        self.logger.info("🏁 Створення профілів...")
        
        # Зберігаємо ВСІ профілі для фінального файлу
        all_profiles = {}
        
        final_pbar = tqdm(supplier_data.items(), desc="Створення профілів")
        profiles_created = 0
        
        for edrpou, items in final_pbar:
            if len(items) > 0:
                try:
                    # ВИКОРИСТОВУЄМО МЕТОД profiler.create_profile()
                    profile = self.profiler.create_profile(edrpou, items)
                    
                    # Зберігаємо в загальний словник
                    all_profiles[edrpou] = profile
                    self.profiler.profiles[edrpou] = profile
                    profiles_created += 1
                        
                except Exception as e:
                    self.logger.error(f"Помилка профілю {edrpou}: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    
            final_pbar.set_postfix({
                'profiles': profiles_created,
                'items': len(items)
            })
        
        # 5. ФІНАЛЬНЕ ЗБЕРЕЖЕННЯ ВСІХ ПРОФІЛІВ
        self.logger.info("💾 Збереження фінального файлу з усіма профілями...")
        
        # Використовуємо вбудований метод profiler для збереження
        self.profiler.save_profiles("supplier_profiles_COMPLETE.json")
        
        self.logger.info(f"✅ ГОТОВО! Створено {len(all_profiles)} профілів")
        self.logger.info(f"📁 Фінальний файл: supplier_profiles_COMPLETE.json")
        
        # 6. Діагностика
        self.logger.info("\n📊 ДІАГНОСТИКА:")
        sorted_suppliers = sorted(supplier_data.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (edrpou, items) in enumerate(sorted_suppliers[:10]):
            profile = all_profiles.get(edrpou)
            if profile:
                self.logger.info(f"   #{i+1} ЄДРПОУ {edrpou}: {len(items)} записів в даних, "
                            f"{profile.metrics.total_positions} позицій в профілі, "
                            f"win_rate={profile.metrics.win_rate:.2%}")
        
        return len(all_profiles)

    
    # def _create_profile(self, edrpou: str, all_items: List[Dict], 
    #              tender_participants: Dict[str, Dict[str, List[Dict]]]) -> SupplierProfile:
    #     """Створення профілю постачальника з усіма метриками"""
    #     from datetime import datetime, timedelta
    #     from collections import Counter, defaultdict
    #     import numpy as np
        
    #     # Основна інформація
    #     name = 'Unknown'
    #     if all_items:
    #         first_item = all_items[0]
    #         name = (first_item.get('supplier_name') or 
    #                 first_item.get('name') or 
    #                 'Unknown')
        
    #     # Унікальні тендери
    #     tender_items = defaultdict(list)
    #     for item in all_items:
    #         tender_num = item.get('tender_number', '')
    #         if tender_num:
    #             tender_items[tender_num].append(item)
        
    #     # Базові метрики
    #     total_positions = len(all_items)
    #     won_positions = sum(1 for item in all_items if item.get('won', False))
    #     total_tenders = len(tender_items)
        
    #     # Визначаємо виграні тендери
    #     won_tenders = 0
    #     for tender_num, items in tender_items.items():
    #         if any(item.get('won', False) for item in items):
    #             won_tenders += 1
        
    #     # ВИЗНАЧАЄМО WIN_RATE ТУТ!
    #     win_rate = won_tenders / total_tenders if total_tenders > 0 else 0
    #     position_win_rate = won_positions / total_positions if total_positions > 0 else 0
        
    #     # Recent metrics
    #     current_date = datetime.now()
    #     recent_date = current_date - timedelta(days=90)
    #     recent_items = []
        
    #     for item in all_items:
    #         date_str = (item.get('date_end') or 
    #                 item.get('date') or 
    #                 item.get('tender_date'))
    #         if date_str:
    #             try:
    #                 if '.' in date_str:
    #                     item_date = datetime.strptime(date_str.split(' ')[0], '%d.%m.%Y')
    #                 else:
    #                     item_date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                    
    #                 if item_date >= recent_date:
    #                     recent_items.append(item)
    #             except:
    #                 pass
        
    #     recent_won = sum(1 for item in recent_items if item.get('won', False))
    #     recent_win_rate = recent_won / len(recent_items) if recent_items else 0
        
    #     # Growth rate
    #     growth_rate = 0
    #     if total_tenders > 10:
    #         sorted_tenders = sorted(tender_items.items(), 
    #                             key=lambda x: x[0])  # Сортуємо по номеру тендера
            
    #         half = len(sorted_tenders) // 2
    #         first_half_won = sum(1 for _, items in sorted_tenders[:half] 
    #                         if any(item.get('won', False) for item in items))
    #         second_half_won = sum(1 for _, items in sorted_tenders[half:] 
    #                             if any(item.get('won', False) for item in items))
            
    #         if first_half_won > 0:
    #             growth_rate = (second_half_won - first_half_won) / first_half_won
    #         elif second_half_won > 0:
    #             growth_rate = 1.0
        
    #     # Статистика по категоріях
    #     category_stats = Counter()
    #     industry_stats = Counter()
    #     cpv_stats = Counter()
    #     brand_stats = Counter()
        
    #     for item in all_items:
    #         # Категорії
    #         category = (item.get('primary_category') or 
    #                 item.get('category') or 
    #                 item.get('item_category'))
    #         if category and category != 'unknown':
    #             category_stats[category] += 1
            
    #         # Індустрії
    #         industry = item.get('industry')
    #         if industry:
    #             industry_stats[industry] += 1
            
    #         # CPV коди
    #         cpv = item.get('cpv') or item.get('cpv_code')
    #         if cpv and cpv != 0:
    #             cpv_str = str(cpv)
    #             if len(cpv_str) >= 8:
    #                 main_cpv = cpv_str[:8]
    #                 cpv_stats[main_cpv] += 1
            
    #         # Бренди
    #         item_name = item.get('item_name') or item.get('tender_name') or ''
    #         if item_name:
    #             known_brands = ['FENDT', 'JOHN DEERE', 'CASE', 'NEW HOLLAND', 'CLAAS',
    #                         'CATERPILLAR', 'KOMATSU', 'VOLVO', 'SCANIA', 'MAN']
    #             for brand in known_brands:
    #                 if brand.lower() in item_name.lower():
    #                     brand_stats[brand] += 1
        
    #     # Конвертуємо в словники
    #     categories = {cat: {'count': count, 'percentage': count/total_positions * 100} 
    #                 for cat, count in category_stats.most_common(10)}
        
    #     industries = {ind: {'count': count, 'percentage': count/total_positions * 100}
    #                 for ind, count in industry_stats.most_common(10)}
        
    #     cpv_experience = {cpv: {'count': count, 'percentage': count/total_positions * 100}
    #                     for cpv, count in cpv_stats.most_common(20)}
        
    #     # Specialization score
    #     specialization_score = 0
    #     if cpv_stats:
    #         top_cpv_count = sum(count for _, count in cpv_stats.most_common(3))
    #         specialization_score = top_cpv_count / total_positions if total_positions > 0 else 0
        
    #     # Stability score
    #     stability_score = 0
    #     if tender_items and len(tender_items) > 1:
    #         dates = []
    #         for tender_num, items in tender_items.items():
    #             date_str = (items[0].get('date_end') or 
    #                     items[0].get('date') or 
    #                     items[0].get('tender_date'))
    #             if date_str:
    #                 try:
    #                     if '.' in date_str:
    #                         date_obj = datetime.strptime(date_str.split(' ')[0], '%d.%m.%Y')
    #                     else:
    #                         date_obj = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
    #                     dates.append(date_obj)
    #                 except:
    #                     pass
            
    #         if len(dates) > 1:
    #             dates.sort()
    #             gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    #             if gaps:
    #                 avg_gap = np.mean(gaps)
    #                 std_gap = np.std(gaps)
    #                 stability_score = 1 - (std_gap / avg_gap) if avg_gap > 0 else 0
    #                 stability_score = max(0, min(1, stability_score))
        
    #     # Competition resistance
    #     competition_resistance = 0
    #     competitions_won = 0
    #     total_competitions = 0
        
    #     for tender_num, participants in tender_participants.items():
    #         if edrpou in participants and len(participants) > 1:
    #             total_competitions += 1
    #             won = any(item.get('won', False) for item in participants[edrpou])
    #             if won:
    #                 competitions_won += 1
        
    #     if total_competitions > 0:
    #         competition_resistance = competitions_won / total_competitions
    #     else:
    #         competition_resistance = win_rate  # Тепер win_rate вже визначений!
        
    #     # Brand expertise
    #     brand_expertise = [{'brand': brand, 'positions': count} 
    #                     for brand, count in brand_stats.most_common(10)]
        
    #     # Переваги та недоліки
    #     competitive_advantages = []
    #     weaknesses = []
        
    #     if win_rate > 0.8:
    #         competitive_advantages.append("Дуже високий відсоток перемог")
    #     if specialization_score > 0.7:
    #         competitive_advantages.append("Вузька спеціалізація")
    #     if stability_score > 0.7:
    #         competitive_advantages.append("Стабільна участь у тендерах")
    #     if competition_resistance > 0.8:
    #         competitive_advantages.append("Високий опір конкуренції")
    #     if growth_rate > 0.2:
    #         competitive_advantages.append("Зростаючий тренд перемог")
        
    #     if win_rate < 0.3:
    #         weaknesses.append("Низький відсоток перемог")
    #     if recent_win_rate < 0.1 and len(recent_items) > 5:
    #         weaknesses.append("Падіння результативності останнім часом")
    #     if specialization_score < 0.3:
    #         weaknesses.append("Занадто широкий профіль без спеціалізації")
    #     if growth_rate < -0.2:
    #         weaknesses.append("Негативний тренд результативності")
        
    #     # Market position
    #     if total_tenders >= 100 and win_rate > 0.7:
    #         market_position = "leader"
    #     elif total_tenders >= 50 and win_rate > 0.5:
    #         market_position = "strong_player"
    #     elif total_tenders >= 20 and win_rate > 0.3:
    #         market_position = "active_participant"
    #     elif total_tenders >= 10:
    #         market_position = "regular_participant"
    #     elif total_tenders >= 5:
    #         market_position = "occasional_participant"
    #     else:
    #         market_position = "new_player"
        
    #     # Reliability score
    #     reliability_score = (
    #         win_rate * 0.3 +
    #         stability_score * 0.2 +
    #         competition_resistance * 0.2 +
    #         (1 - abs(growth_rate)) * 0.1 +
    #         min(total_tenders / 100, 1) * 0.2
    #     )
        
    #     # Створюємо профіль
    #     return SupplierProfile(
    #         edrpou=edrpou,
    #         name=name,
    #         metrics=SupplierMetrics(
    #             total_tenders=total_tenders,
    #             won_tenders=won_tenders,
    #             total_positions=total_positions,
    #             won_positions=won_positions,
    #             win_rate=win_rate,
    #             position_win_rate=position_win_rate,
    #             recent_win_rate=recent_win_rate,
    #             growth_rate=growth_rate,
    #             stability_score=stability_score,
    #             specialization_score=specialization_score,
    #             competition_resistance=competition_resistance
    #         ),
    #         categories=categories,
    #         industries=industries,
    #         cpv_experience=cpv_experience,
    #         brand_expertise=brand_expertise,
    #         competitive_advantages=competitive_advantages,
    #         weaknesses=weaknesses,
    #         market_position=market_position,
    #         reliability_score=reliability_score
    #     )


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