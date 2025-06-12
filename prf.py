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
        
    def emergency_load_and_build(self):
        """АВАРІЙНИЙ РЕЖИМ - завантаження і побудова одночасно"""
        self.logger.info("🚨 АВАРІЙНИЙ РЕЖИМ ПОБУДОВИ")
        
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
            
            # Перевірка для конкретного ЄДРПОУ
            for edrpou, items in list(supplier_data.items())[:5]:  # Перші 5 для діагностики
                self.logger.info(f"   ЄДРПОУ {edrpou}: {len(items)} записів")

            # Зберігаємо кеш даних
            self.logger.info("💾 Зберігаємо кеш даних...")
            with open(cache_file, 'wb') as f:
                pickle.dump(dict(supplier_data), f)

        # 4. Побудова індексу: tender_number -> {edrpou: [items]}
        tender_participants = defaultdict(lambda: defaultdict(list))
        for edrpou, items in supplier_data.items():
            for item in items:
                tender = item.get('tender_number', '')
                if tender:
                    tender_participants[tender][edrpou].append(item)

        # 5. Створення профілів
        self.logger.info("🏁 Створення профілів...")
        final_pbar = tqdm(supplier_data.items(), desc="Створення профілів")
        batch_for_save = {}
        profiles_created = 0
        
        for edrpou, items in final_pbar:
            if len(items) > 0:  # Видалив перевірку на існування в профайлері
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
                    self.logger.error(f"Помилка профілю {edrpou}: {e}")
                    
            final_pbar.set_postfix({
                'profiles': profiles_created,
                'items': len(items)  # Показуємо кількість items
            })

        # Зберігаємо залишок
        if batch_for_save:
            self._save_batch(batch_for_save, profiles_created)

        # 6. Фінальне збереження
        self.profiler.save_profiles("supplier_profiles_COMPLETE.json")
        self.logger.info(f"✅ ГОТОВО! Створено {len(self.profiler.profiles)} профілів")
        
        # Діагностика
        self.logger.info("\n📊 ДІАГНОСТИКА:")
        # Вибираємо кілька постачальників з різною кількістю записів
        sorted_suppliers = sorted(supplier_data.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (edrpou, items) in enumerate(sorted_suppliers[:10]):
            profile = self.profiler.profiles.get(edrpou)
            if profile:
                self.logger.info(f"   #{i+1} ЄДРПОУ {edrpou}: {len(items)} записів в даних, "
                               f"{profile.metrics.total_positions} позицій в профілі")
        
        return len(self.profiler.profiles)

    
    def _create_profile(self, edrpou: str, all_items: List[Dict], 
                     tender_participants: Dict[str, Dict[str, List[Dict]]]) -> SupplierProfile:
        """Створення профілю постачальника з усіма метриками БЕЗ звернення до інших профілів"""
        from datetime import datetime, timedelta
        from collections import Counter, defaultdict
        import numpy as np
        
        # Основна інформація - шукаємо правильне поле з назвою
        name = 'Unknown'
        if all_items:
            # Пробуємо різні можливі поля з назвою постачальника
            first_item = all_items[0]
            name = (first_item.get('supplier_name') or 
                    first_item.get('name') or 
                    'Unknown')
        
        # Унікальні тендери
        tender_items = defaultdict(list)
        for item in all_items:
            tender_num = item.get('tender_number', '')
            if tender_num:
                tender_items[tender_num].append(item)
        
        # Метрики по тендерах
        won_tenders = 0
        total_positions = len(all_items)
        won_positions = sum(1 for item in all_items if item.get('won', False))
        
        # Визначаємо виграні тендери
        for tender_num, items in tender_items.items():
            if any(item.get('won', False) for item in items):
                won_tenders += 1
        
        total_tenders = len(tender_items)
        
        # ВИПРАВЛЕННЯ 1: Recent metrics (останні 90 днів)
        current_date = datetime.now()
        recent_date = current_date - timedelta(days=90)
        recent_items = []
        
        for item in all_items:
            # Перевіряємо різні формати дати
            date_str = (item.get('date_end') or 
                    item.get('date') or 
                    item.get('tender_date') or 
                    item.get('announcement_date') or 
                    item.get('extraction_date'))
            if date_str:
                try:
                    # Пробуємо різні формати дат
                    for fmt in ['%d.%m.%Y', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                        try:
                            if '.' in date_str and fmt.startswith('%d'):
                                item_date = datetime.strptime(date_str.split(' ')[0], fmt)
                            else:
                                item_date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                            
                            if item_date >= recent_date:
                                recent_items.append(item)
                            break
                        except:
                            continue
                except:
                    pass
        
        recent_won = sum(1 for item in recent_items if item.get('won', False))
        recent_win_rate = recent_won / len(recent_items) if recent_items else 0
        
        # ВИПРАВЛЕННЯ 2: Growth rate (порівняння періодів)
        growth_rate = 0
        if total_tenders > 10:  # Достатньо даних для аналізу
            # Розділяємо на дві половини за часом
            sorted_items = sorted(tender_items.items(), 
                                key=lambda x: min(item.get('date', '') for item in x[1]))
            
            half = len(sorted_items) // 2
            first_half_won = sum(1 for _, items in sorted_items[:half] 
                            if any(item.get('won', False) for item in items))
            second_half_won = sum(1 for _, items in sorted_items[half:] 
                                if any(item.get('won', False) for item in items))
            
            if first_half_won > 0:
                growth_rate = (second_half_won - first_half_won) / first_half_won
            elif second_half_won > 0:
                growth_rate = 1.0  # 100% growth from 0
        
        # ВИПРАВЛЕННЯ 3: Category, Industry, CPV статистика
        category_stats = Counter()
        industry_stats = Counter()
        cpv_stats = Counter()
        brand_stats = Counter()
        
        for item in all_items:
            # Категорії - перевіряємо різні можливі поля
            category = (item.get('primary_category') or 
                    item.get('category') or 
                    item.get('item_category'))
            if category and category != 'unknown':
                category_stats[category] += 1
            
            # Індустрії  
            industry = item.get('industry')
            if industry:
                industry_stats[industry] += 1
                
            # CPV коди
            cpv = item.get('cpv') or item.get('cpv_code')
            if cpv and cpv != 0:
                # Конвертуємо в string і беремо основний код
                cpv_str = str(cpv)
                if len(cpv_str) >= 8:
                    main_cpv = cpv_str[:8]
                    cpv_stats[main_cpv] += 1
                
            # Бренди/виробники - шукаємо в назві товару
            item_name = item.get('item_name') or item.get('tender_name') or ''
            if item_name:
                # Простий пошук відомих брендів
                known_brands = ['FENDT', 'JOHN DEERE', 'CASE', 'NEW HOLLAND', 'CLAAS',
                            'CATERPILLAR', 'KOMATSU', 'VOLVO', 'SCANIA', 'MAN']
                for brand in known_brands:
                    if brand.lower() in item_name.lower():
                        brand_stats[brand] += 1
        
        # Перетворюємо Counter в словники з відсотками
        categories = {cat: {'count': count, 'percentage': count/total_positions * 100} 
                    for cat, count in category_stats.most_common(10)}
        
        industries = {ind: {'count': count, 'percentage': count/total_positions * 100}
                    for ind, count in industry_stats.most_common(10)}
        
        cpv_experience = {cpv: {'count': count, 'percentage': count/total_positions * 100}
                        for cpv, count in cpv_stats.most_common(20)}
        
        # ВИПРАВЛЕННЯ 4: Specialization score
        specialization_score = 0
        if cpv_stats:
            # Чим більше концентрація в топ CPV, тим вище спеціалізація
            top_cpv_count = sum(count for _, count in cpv_stats.most_common(3))
            specialization_score = top_cpv_count / total_positions if total_positions > 0 else 0
        
        # ВИПРАВЛЕННЯ 5: Stability score (стабільність участі)
        stability_score = 0
        if tender_items:
            # Аналізуємо розподіл тендерів за часом
            dates = []
            for tender_num, items in tender_items.items():
                # Беремо дату з першого елементу тендера
                date_str = (items[0].get('date_end') or 
                        items[0].get('date') or 
                        items[0].get('tender_date'))
                if date_str:
                    try:
                        if '.' in date_str:
                            date_obj = datetime.strptime(date_str.split(' ')[0], '%d.%m.%Y')
                        else:
                            date_obj = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d')
                        dates.append(date_obj)
                    except:
                        pass
            
            if len(dates) > 1:
                dates.sort()
                # Середня кількість днів між тендерами
                gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                if gaps:
                    avg_gap = np.mean(gaps)
                    std_gap = np.std(gaps)
                    # Чим менше стандартне відхилення відносно середнього, тим стабільніше
                    stability_score = 1 - (std_gap / avg_gap) if avg_gap > 0 else 0
                    stability_score = max(0, min(1, stability_score))  # Обмежуємо 0-1
        
        # ВИПРАВЛЕННЯ 6: Competition resistance (стійкість до конкуренції)
        # Спрощена версія без звернення до інших профілів
        competition_resistance = 0
        competitions_won = 0
        total_competitions = 0
        
        for tender_num, participants in tender_participants.items():
            if edrpou in participants and len(participants) > 1:
                total_competitions += 1
                # Чи виграв цей постачальник?
                won = any(item.get('won', False) for item in participants[edrpou])
                if won:
                    competitions_won += 1
        
        if total_competitions > 0:
            # Базова метрика: відсоток виграних тендерів де була конкуренція
            competition_resistance = competitions_won / total_competitions
        else:
            # Якщо немає конкурентних тендерів, беремо загальний win rate
            competition_resistance = win_rate
        
        # ВИПРАВЛЕННЯ 7: Brand expertise
        brand_expertise = [{'brand': brand, 'positions': count} 
                        for brand, count in brand_stats.most_common(10)]
        
        # ВИПРАВЛЕННЯ 8: Competitive advantages & weaknesses
        competitive_advantages = []
        weaknesses = []
        
        win_rate = won_tenders / total_tenders if total_tenders > 0 else 0
        
        # Аналіз переваг
        if win_rate > 0.8:
            competitive_advantages.append("Дуже високий відсоток перемог")
        if specialization_score > 0.7:
            competitive_advantages.append("Вузька спеціалізація")
        if stability_score > 0.7:
            competitive_advantages.append("Стабільна участь у тендерах")
        if competition_resistance > 0.8:
            competitive_advantages.append("Високий опір конкуренції")
        if growth_rate > 0.2:
            competitive_advantages.append("Зростаючий тренд перемог")
        
        # Аналіз слабкостей
        if win_rate < 0.3:
            weaknesses.append("Низький відсоток перемог")
        if recent_win_rate < 0.1 and len(recent_items) > 5:
            weaknesses.append("Падіння результативності останнім часом")
        if specialization_score < 0.3:
            weaknesses.append("Занадто широкий профіль без спеціалізації")
        if growth_rate < -0.2:
            weaknesses.append("Негативний тренд результативності")
        
        # ВИПРАВЛЕННЯ 9: Market position
        market_position = "unknown"
        if total_tenders >= 100 and win_rate > 0.7:
            market_position = "leader"
        elif total_tenders >= 50 and win_rate > 0.5:
            market_position = "strong_player"
        elif total_tenders >= 20 and win_rate > 0.3:
            market_position = "active_participant"
        elif total_tenders >= 10:
            market_position = "regular_participant"
        elif total_tenders >= 5:
            market_position = "occasional_participant"
        else:
            market_position = "new_player"
        
        # Reliability score
        reliability_score = (
            win_rate * 0.3 +
            stability_score * 0.2 +
            competition_resistance * 0.2 +
            (1 - abs(growth_rate)) * 0.1 +  # Стабільність краще різких змін
            min(total_tenders / 100, 1) * 0.2  # Досвід
        )
        
        # Створюємо профіль
        return SupplierProfile(
            edrpou=edrpou,
            name=name,
            metrics=SupplierMetrics(
                total_tenders=total_tenders,
                won_tenders=won_tenders,
                total_positions=total_positions,
                won_positions=won_positions,
                win_rate=win_rate,
                position_win_rate=won_positions / total_positions if total_positions > 0 else 0,
                recent_win_rate=recent_win_rate,
                growth_rate=growth_rate,
                stability_score=stability_score,
                specialization_score=specialization_score,
                competition_resistance=competition_resistance
            ),
            categories=categories,
            industries=industries,
            cpv_experience=cpv_experience,
            brand_expertise=brand_expertise,
            competitive_advantages=competitive_advantages,
            weaknesses=weaknesses,
            market_position=market_position,
            reliability_score=reliability_score
        )








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