# # build_profiles.py
# import json
# import logging
# from datetime import datetime
# from collections import defaultdict
# from typing import Dict, List, Any
# from tqdm import tqdm
# import numpy as np
# from pathlib import Path
# import gc

# from tender_analysis_system import TenderAnalysisSystem
# from supplier_profiler import SupplierProfile, SupplierMetrics

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class FastProfileBuilder:
#     """Оптимізований побудовник профілів постачальників"""
    
#     def __init__(self, system: TenderAnalysisSystem):
#         self.system = system
#         self.vector_db = system.vector_db
#         self.profiler = system.supplier_profiler
#         self.categories_manager = system.categories_manager
        
#         # Кеші для оптимізації
#         self.category_cache = {}
#         self.brand_patterns = self.profiler.brand_patterns
        
#     def build_all_profiles_optimized(self, batch_size: int = 10000, save_every: int = 5000):
#         """
#         Оптимізована побудова профілів з батчевою обробкою
#         """
#         logger.info("🚀 Початок оптимізованої побудови профілів")
#         start_time = datetime.now()
        
#         # 1. Отримання всіх унікальних EDRPOU
#         logger.info("📊 Отримання списку постачальників...")
#         all_edrpou = self._get_all_suppliers_fast()
#         logger.info(f"✅ Знайдено {len(all_edrpou):,} унікальних постачальників")
        
#         if not all_edrpou:
#             logger.error("❌ Не знайдено постачальників!")
#             return
        
#         # 2. Розбиття на батчі
#         total_batches = (len(all_edrpou) + batch_size - 1) // batch_size
#         logger.info(f"📦 Розбито на {total_batches} батчів по {batch_size} постачальників")
        
#         profiles_created = 0
#         profiles_batch = {}
        
#         # 3. Обробка батчами
#         for batch_idx in range(0, len(all_edrpou), batch_size):
#             batch_edrpou = all_edrpou[batch_idx:batch_idx + batch_size]
#             batch_num = batch_idx // batch_size + 1
            
#             logger.info(f"\n{'='*60}")
#             logger.info(f"🔄 Обробка батчу {batch_num}/{total_batches} ({len(batch_edrpou)} постачальників)")
            
#             # Завантаження даних для всього батчу одразу
#             batch_data = self._load_batch_data(batch_edrpou)
            
#             # Створення профілів для батчу
#             pbar = tqdm(batch_edrpou, desc=f"Батч {batch_num}", unit="профілів")
            
#             for edrpou in pbar:
#                 if edrpou in batch_data and batch_data[edrpou]:
#                     try:
#                         profile = self._create_profile_fast(edrpou, batch_data[edrpou])
#                         if profile:
#                             profiles_batch[edrpou] = profile
#                             profiles_created += 1
                            
#                             # Періодичне збереження
#                             if profiles_created % save_every == 0:
#                                 self._save_profiles_batch(profiles_batch)
#                                 profiles_batch.clear()
#                                 gc.collect()
                                
#                     except Exception as e:
#                         logger.error(f"Помилка створення профілю для {edrpou}: {e}")
                
#                 pbar.set_postfix({
#                     'created': profiles_created,
#                     'memory_mb': self._get_memory_usage()
#                 })
            
#             # Статистика батчу
#             batch_time = (datetime.now() - start_time).total_seconds()
#             speed = profiles_created / batch_time if batch_time > 0 else 0
#             logger.info(f"✅ Батч {batch_num} завершено. Швидкість: {speed:.1f} профілів/сек")
            
#             # Очищення пам'яті між батчами
#             gc.collect()
        
#         # 4. Збереження останніх профілів
#         if profiles_batch:
#             self._save_profiles_batch(profiles_batch)
        
#         # 5. Фінальна статистика
#         total_time = (datetime.now() - start_time).total_seconds()
#         logger.info("\n" + "="*60)
#         logger.info("✅ ПОБУДОВА ПРОФІЛІВ ЗАВЕРШЕНА")
#         logger.info(f"📊 Створено профілів: {profiles_created:,}")
#         logger.info(f"⏱️ Загальний час: {total_time:.1f} сек ({total_time/60:.1f} хв)")
#         logger.info(f"🚀 Середня швидкість: {profiles_created/total_time:.1f} профілів/сек")
#         logger.info("="*60)
        
#         # 6. Збереження фінального файлу
#         self.profiler.save_profiles("supplier_profiles_complete.json")
        
#         return {
#             'total_suppliers': len(all_edrpou),
#             'profiles_created': profiles_created,
#             'processing_time': total_time
#         }
    
#     def _get_all_suppliers_fast(self) -> List[str]:
#         """Швидке отримання всіх EDRPOU через scroll API"""
#         edrpou_set = set()
#         offset = None
#         batch_size = 10000
        
#         logger.info("📥 Завантаження EDRPOU з векторної бази...")
        
#         while True:
#             try:
#                 # Використовуємо scroll для швидкого проходу
#                 records, next_offset = self.vector_db.client.scroll(
#                     collection_name=self.vector_db.collection_name,
#                     offset=offset,
#                     limit=batch_size,
#                     with_payload=["edrpou"],  # Тільки EDRPOU
#                     with_vectors=False  # Без векторів
#                 )
                
#                 for record in records:
#                     if record.payload and 'edrpou' in record.payload:
#                         edrpou_set.add(record.payload['edrpou'])
                
#                 if not next_offset:
#                     break
#                 offset = next_offset
                
#                 if len(edrpou_set) % 50000 == 0:
#                     logger.info(f"   Завантажено {len(edrpou_set):,} EDRPOU...")
                    
#             except Exception as e:
#                 logger.error(f"Помилка завантаження EDRPOU: {e}")
#                 break
        
#         return list(edrpou_set)
    
#     def _load_batch_data(self, edrpou_list: List[str]) -> Dict[str, List[Dict]]:
#         """ШВИДКЕ завантаження даних для батчу постачальників"""
#         batch_data = defaultdict(list)
        
#         logger.info(f"⚡ Швидке завантаження даних для {len(edrpou_list)} постачальників...")
        
#         # ВАЖЛИВО: Завантажуємо ВСІ дані одним запитом через scroll
#         # без фільтрації на стороні Qdrant
        
#         offset = None
#         total_loaded = 0
        
#         # Створюємо set для швидкої перевірки
#         edrpou_set = set(edrpou_list)
        
#         while True:
#             try:
#                 # Простий scroll БЕЗ фільтрів
#                 records, next_offset = self.vector_db.client.scroll(
#                     collection_name=self.vector_db.collection_name,
#                     offset=offset,
#                     limit=10000,  # Великий батч
#                     with_payload=True,
#                     with_vectors=False
#                 )
                
#                 if not records:
#                     break
                
#                 # Фільтруємо на стороні Python (набагато швидше!)
#                 for record in records:
#                     if record.payload and 'edrpou' in record.payload:
#                         edrpou = record.payload['edrpou']
#                         if edrpou in edrpou_set:
#                             batch_data[edrpou].append(record.payload)
#                             total_loaded += 1
                
#                 # Якщо вже завантажили всі потрібні - виходимо
#                 if len(batch_data) >= len(edrpou_list):
#                     break
                    
#                 if not next_offset:
#                     break
#                 offset = next_offset
                
#             except Exception as e:
#                 logger.error(f"Помилка завантаження: {e}")
#                 break
        
#         logger.info(f"✅ Завантажено {total_loaded} записів для {len(batch_data)} постачальників")
        
#         return batch_data

    
#     def _safe_parse_date(self, date_str: str) -> bool:
#         """Безпечний парсинг дати"""
#         try:
#             datetime.strptime(date_str, '%d.%m.%Y')
#             return True
#         except:
#             return False
    
#     def _calculate_reliability(self, metrics, stability: float, category_count: int) -> float:
#         """Розрахунок надійності"""
#         factors = []
        
#         # Досвід
#         experience_score = min(metrics.total_tenders / 100, 1.0)
#         factors.append(experience_score * 0.3)
        
#         # Ефективність
#         win_rate_score = min(metrics.win_rate * 2, 1.0)
#         factors.append(win_rate_score * 0.3)
        
#         # Стабільність
#         factors.append(stability * 0.2)
        
#         # Диверсифікація
#         diversity_score = min(category_count / 5, 1.0)
#         factors.append(diversity_score * 0.2)
        
#         return sum(factors)

    
#     def _identify_weaknesses(self, metrics, recent_stats, categories) -> List[str]:
#         """Визначення слабких сторін"""
#         weaknesses = []
        
#         if metrics.win_rate < 0.15:
#             weaknesses.append(f"Низький win rate: {metrics.win_rate:.1%}")
        
#         if metrics.growth_rate < -0.2:
#             weaknesses.append("Спад активності")
        
#         if metrics.specialization_score < 0.2 and len(categories) > 5:
#             weaknesses.append("Відсутність чіткої спеціалізації")
        
#         if recent_stats.get('recent_total', 0) < 5:
#             weaknesses.append("Низька недавня активність")
        
#         return weaknesses



    
#     def _identify_advantages(self, metrics, categories, brands, quality_levels) -> List[str]:
#         """Визначення конкурентних переваг"""
#         advantages = []
        
#         if metrics.win_rate >= 0.35:
#             advantages.append(f"Високий win rate: {metrics.win_rate:.1%}")
        
#         if metrics.stability_score >= 0.75:
#             advantages.append("Стабільна діяльність")
        
#         if metrics.specialization_score >= 0.6:
#             top_category = max(categories.items(), key=lambda x: x[1]['total'])[0]
#             advantages.append(f"Спеціалізація: {top_category}")
        
#         if brands:
#             advantages.append(f"Досвід з брендами: {', '.join(brands[:3])}")
        
#         if metrics.recent_win_rate > metrics.win_rate * 1.2:
#             advantages.append("Позитивна динаміка")
        
#         if quality_levels['premium'] > quality_levels['budget'] * 2:
#             advantages.append("Фокус на преміум сегменті")
        
#         return advantages
    
#     def _determine_market_position(self, total_tenders: int, win_rate: float, 
#                               total_budget: float, stability: float) -> str:
#         """Визначення позиції на ринку"""
#         score = 0
        
#         # Фактори оцінки
#         if total_tenders >= 100:
#             score += 3
#         elif total_tenders >= 50:
#             score += 2
#         elif total_tenders >= 20:
#             score += 1
        
#         if win_rate >= 0.4:
#             score += 2
#         elif win_rate >= 0.25:
#             score += 1
        
#         if total_budget >= 10_000_000:
#             score += 2
#         elif total_budget >= 1_000_000:
#             score += 1
        
#         if stability >= 0.7:
#             score += 1
        
#         # Класифікація
#         if score >= 7:
#             return "market_leader"
#         elif score >= 5:
#             return "established_player"
#         elif score >= 3:
#             return "competitive_player"
#         elif score >= 1:
#             return "emerging_player"
#         else:
#             return "new_entrant"
    
#     def _calculate_competition_resistance(self, tender_stats: Dict, overall_win_rate: float) -> float:
#         """Оцінка стійкості до конкуренції"""
#         # Аналіз перемог в тендерах з багатьма учасниками
#         high_competition_wins = 0
#         high_competition_total = 0
        
#         for stat in tender_stats.values():
#             # Припускаємо високу конкуренцію якщо багато позицій
#             if stat['total_positions'] > 5:
#                 high_competition_total += 1
#                 if stat['won_positions'] > 0:
#                     high_competition_wins += 1
        
#         if high_competition_total > 0:
#             high_comp_win_rate = high_competition_wins / high_competition_total
#             # Порівнюємо з загальним win rate
#             resistance = high_comp_win_rate / (overall_win_rate + 0.001)
#             return min(resistance, 1.0)
        
#         return 0.5  # Невідомо

    
#     def _calculate_stability(self, tender_stats: Dict, position_win_rates: List[float]) -> float:
#         """Розрахунок стабільності постачальника"""
#         factors = []
        
#         # 1. Стабільність win rate
#         if len(position_win_rates) >= 2:
#             variance = np.var(position_win_rates)
#             stability_from_variance = 1.0 - min(variance * 2, 1.0)
#             factors.append(stability_from_variance)
        
#         # 2. Регулярність участі (наскільки рівномірно розподілена активність)
#         if len(tender_stats) >= 4:
#             # Аналіз по місяцях
#             monthly_activity = defaultdict(int)
#             for stat in tender_stats.values():
#                 for date_str in stat['dates']:
#                     try:
#                         date = datetime.strptime(date_str, '%d.%m.%Y')
#                         month_key = date.strftime('%Y-%m')
#                         monthly_activity[month_key] += 1
#                     except:
#                         pass
            
#             if len(monthly_activity) >= 3:
#                 activity_values = list(monthly_activity.values())
#                 regularity = 1.0 - (np.std(activity_values) / (np.mean(activity_values) + 0.001))
#                 factors.append(max(0, min(1, regularity)))
        
#         # 3. Досвід
#         experience_factor = min(len(tender_stats) / 50, 1.0)
#         factors.append(experience_factor)
        
#         return np.mean(factors) if factors else 0.5



#     def _calculate_recent_stats(self, tender_stats: Dict) -> Dict:
#         """Розрахунок недавньої статистики (6 місяців)"""
#         from datetime import datetime, timedelta
        
#         # Збираємо всі дати
#         all_dates = []
#         for stat in tender_stats.values():
#             for date_str in stat['dates']:
#                 try:
#                     date = datetime.strptime(date_str, '%d.%m.%Y')
#                     all_dates.append(date)
#                 except:
#                     pass
        
#         if not all_dates:
#             return {'win_rate': 0.0, 'growth_rate': 0.0}
        
#         # Визначаємо період
#         latest_date = max(all_dates)
#         cutoff_date = latest_date - timedelta(days=180)
        
#         # Фільтруємо недавні тендери
#         recent_won = 0
#         recent_total = 0
        
#         for tender_num, stat in tender_stats.items():
#             tender_dates = []
#             for date_str in stat['dates']:
#                 try:
#                     date = datetime.strptime(date_str, '%d.%m.%Y')
#                     tender_dates.append(date)
#                 except:
#                     pass
            
#             if tender_dates and max(tender_dates) >= cutoff_date:
#                 recent_total += 1
#                 if stat['won_positions'] > 0:
#                     recent_won += 1
        
#         recent_win_rate = recent_won / recent_total if recent_total > 0 else 0.0
        
#         # Growth rate (порівняння з попереднім періодом)
#         old_cutoff = cutoff_date - timedelta(days=180)
#         old_total = sum(1 for stat in tender_stats.values() 
#                     if any(old_cutoff <= datetime.strptime(d, '%d.%m.%Y') < cutoff_date 
#                             for d in stat['dates'] if self._safe_parse_date(d)))
        
#         growth_rate = (recent_total - old_total) / old_total if old_total > 0 else 0.0
        
#         return {
#             'win_rate': recent_win_rate,
#             'growth_rate': growth_rate,
#             'recent_total': recent_total,
#             'recent_won': recent_won
#         }




#     def _create_profile_fast(self, edrpou: str, items: List[Dict]) -> SupplierProfile:
#         """
#         Швидке створення профілю з детальними метриками
        
#         Args:
#             edrpou: код ЄДРПОУ постачальника
#             items: список записів (позицій) постачальника
#         """
#         if not items:
#             return None
        
#         # Базова інформація
#         first_item = items[0]
#         profile = SupplierProfile(
#             edrpou=edrpou,
#             name=first_item.get('supplier_name', '') or first_item.get('supp_name', '')
#         )
        
#         # Ініціалізація метрик
#         metrics = profile.metrics
        
#         # Структури для збору статистики
#         tender_stats = defaultdict(lambda: {
#             'total_positions': 0,
#             'won_positions': 0,
#             'total_budget': 0.0,
#             'won_budget': 0.0,
#             'dates': [],
#             'industries': set(),
#             'categories': set()
#         })
        
#         category_stats = defaultdict(lambda: {
#             'total': 0,
#             'won': 0,
#             'revenue': 0.0,
#             'tenders': set(),
#             'items': []
#         })
        
#         industry_stats = defaultdict(lambda: {
#             'total': 0,
#             'won': 0,
#             'revenue': 0.0,
#             'tenders': set()
#         })
        
#         cpv_stats = defaultdict(lambda: {
#             'total': 0,
#             'won': 0,
#             'categories': set()
#         })
        
#         # Для аналізу брендів та якості
#         brand_counts = defaultdict(int)
#         quality_levels = {'premium': 0, 'standard': 0, 'budget': 0}
        
#         # Обробка кожної позиції
#         for item in items:
#             tender_num = item.get('tender_number', '')
#             is_won = bool(item.get('won', False))
            
#             # Базові підрахунки
#             metrics.total_positions += 1
#             if is_won:
#                 metrics.won_positions += 1
            
#             # Статистика по тендерах
#             if tender_num:
#                 tender_stat = tender_stats[tender_num]
#                 tender_stat['total_positions'] += 1
#                 if is_won:
#                     tender_stat['won_positions'] += 1
                
#                 # Бюджет
#                 try:
#                     budget = float(item.get('budget', 0) or 0)
#                     tender_stat['total_budget'] += budget
#                     if is_won:
#                         tender_stat['won_budget'] += budget
#                 except:
#                     pass
                
#                 # Дата для аналізу трендів
#                 date_str = item.get('date_end', '')
#                 if date_str:
#                     tender_stat['dates'].append(date_str)
                
#                 # Індустрія
#                 industry = item.get('industry', 'unknown')
#                 if industry:
#                     tender_stat['industries'].add(industry)
            
#             # Категорійна статистика
#             category = item.get('primary_category', 'unknown')
#             if category == 'unknown' and self.categories_manager:
#                 item_name = item.get('item_name', '')
#                 if item_name:
#                     category = self._get_cached_category(item_name)
            
#             cat_stat = category_stats[category]
#             cat_stat['total'] += 1
#             if is_won:
#                 cat_stat['won'] += 1
#                 try:
#                     revenue = float(item.get('budget', 0) or 0)
#                     cat_stat['revenue'] += revenue
#                 except:
#                     pass
#             if tender_num:
#                 cat_stat['tenders'].add(tender_num)
            
#             # Збираємо назви товарів для аналізу
#             item_name = item.get('item_name', '')
#             if item_name and len(cat_stat['items']) < 10:  # Обмежуємо для економії пам'яті
#                 cat_stat['items'].append(item_name)
            
#             # Індустрійна статистика
#             industry = item.get('industry', 'unknown')
#             ind_stat = industry_stats[industry]
#             ind_stat['total'] += 1
#             if is_won:
#                 ind_stat['won'] += 1
#                 try:
#                     revenue = float(item.get('budget', 0) or 0)
#                     ind_stat['revenue'] += revenue
#                 except:
#                     pass
#             if tender_num:
#                 ind_stat['tenders'].add(tender_num)
            
#             # CPV статистика
#             cpv = item.get('cpv', 0)
#             if cpv:
#                 cpv_stat = cpv_stats[str(cpv)]
#                 cpv_stat['total'] += 1
#                 if is_won:
#                     cpv_stat['won'] += 1
#                 cpv_stat['categories'].add(category)
            
#             # Аналіз брендів (якщо виграна позиція)
#             if is_won and item_name:
#                 for brand, pattern in self.brand_patterns.items():
#                     if pattern.search(item_name):
#                         brand_counts[brand] += 1
            
#             # Аналіз рівня якості
#             if item_name:
#                 item_lower = item_name.lower()
#                 if any(ind in item_lower for ind in ['оригінал', 'преміум', 'високоякіс']):
#                     quality_levels['premium'] += 1
#                 elif any(ind in item_lower for ind in ['економ', 'бюджет', 'аналог']):
#                     quality_levels['budget'] += 1
#                 else:
#                     quality_levels['standard'] += 1
        
#         # === ФІНАЛІЗАЦІЯ МЕТРИК ===
        
#         # Метрики тендерів
#         metrics.total_tenders = len(tender_stats)
        
#         # Різні способи підрахунку "виграних" тендерів
#         metrics.won_tenders = sum(1 for stat in tender_stats.values() 
#                                 if stat['won_positions'] > 0)  # Виграв хоча б щось
        
#         # Додаткові метрики тендерів
#         fully_won = sum(1 for stat in tender_stats.values() 
#                     if stat['won_positions'] == stat['total_positions'])
        
#         majority_won = sum(1 for stat in tender_stats.values() 
#                         if stat['won_positions'] > stat['total_positions'] / 2)
        
#         # Win rates
#         if metrics.total_tenders > 0:
#             metrics.win_rate = metrics.won_tenders / metrics.total_tenders
        
#         if metrics.total_positions > 0:
#             metrics.position_win_rate = metrics.won_positions / metrics.total_positions
        
#         # Середній % виграних позицій в тендерах де брав участь
#         position_win_rates = []
#         for stat in tender_stats.values():
#             if stat['total_positions'] > 0:
#                 position_win_rates.append(stat['won_positions'] / stat['total_positions'])
        
#         if position_win_rates:
#             avg_tender_win_rate = np.mean(position_win_rates)
#         else:
#             avg_tender_win_rate = 0.0
        
#         # Фінансові метрики
#         total_budget = sum(stat['total_budget'] for stat in tender_stats.values())
#         won_budget = sum(stat['won_budget'] for stat in tender_stats.values())
        
#         # Recent performance (останні 6 місяців)
#         recent_stats = self._calculate_recent_stats(tender_stats)
#         metrics.recent_win_rate = recent_stats.get('win_rate', 0.0)
#         metrics.growth_rate = recent_stats.get('growth_rate', 0.0)
        
#         # Stability score
#         metrics.stability_score = self._calculate_stability(tender_stats, position_win_rates)
        
#         # Competition resistance (наскільки добре виграє в конкурентних тендерах)
#         metrics.competition_resistance = self._calculate_competition_resistance(
#             tender_stats, metrics.position_win_rate
#         )
        
#         # === ЗБЕРЕЖЕННЯ В ПРОФІЛЬ ===
        
#         # Категорії
#         for category, stat in category_stats.items():
#             if stat['total'] > 0:
#                 profile.categories[category] = {
#                     'total': stat['total'],
#                     'won': stat['won'],
#                     'revenue': stat['revenue'],
#                     'win_rate': stat['won'] / stat['total'],
#                     'tender_participation': len(stat['tenders']),
#                     'specialization': stat['total'] / metrics.total_positions if metrics.total_positions > 0 else 0
#                 }
        
#         # Індустрії
#         for industry, stat in industry_stats.items():
#             if stat['total'] > 0:
#                 profile.industries[industry] = {
#                     'total': stat['total'],
#                     'won': stat['won'],
#                     'revenue': stat['revenue'],
#                     'win_rate': stat['won'] / stat['total'],
#                     'tender_participation': len(stat['tenders'])
#                 }
        
#         # CPV досвід (топ-10)
#         top_cpv = sorted(cpv_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
#         for cpv, stat in top_cpv:
#             if stat['total'] > 0:
#                 profile.cpv_experience[cpv] = {
#                     'total': stat['total'],
#                     'won': stat['won'],
#                     'win_rate': stat['won'] / stat['total'] if stat['total'] > 0 else 0,
#                     'categories': list(stat['categories'])
#                 }
        
#         # Бренди (топ-5)
#         if brand_counts:
#             profile.brand_expertise = [brand for brand, _ in 
#                                     sorted(brand_counts.items(), 
#                                             key=lambda x: x[1], reverse=True)[:5]]
        
#         # Спеціалізація
#         if profile.categories:
#             category_shares = [cat['specialization'] for cat in profile.categories.values()]
#             metrics.specialization_score = sum(s**2 for s in category_shares)  # HHI
        
#         # Ринкова позиція
#         profile.market_position = self._determine_market_position(
#             metrics.total_tenders,
#             metrics.win_rate,
#             total_budget,
#             metrics.stability_score
#         )
        
#         # Конкурентні переваги та слабкості
#         profile.competitive_advantages = self._identify_advantages(
#             metrics, profile.categories, profile.brand_expertise, quality_levels
#         )
        
#         profile.weaknesses = self._identify_weaknesses(
#             metrics, recent_stats, profile.categories
#         )
        
#         # Надійність
#         profile.reliability_score = self._calculate_reliability(
#             metrics, metrics.stability_score, len(profile.categories)
#         )
        
#         # Додаткові метрики в словник (для розширеного аналізу)
#         profile.extended_metrics = {
#             'fully_won_tenders': fully_won,
#             'majority_won_tenders': majority_won,
#             'avg_tender_position_win_rate': avg_tender_win_rate,
#             'total_budget_participated': total_budget,
#             'total_budget_won': won_budget,
#             'budget_win_rate': won_budget / total_budget if total_budget > 0 else 0,
#             'quality_distribution': dict(quality_levels),
#             'unique_cpv_codes': len(cpv_stats),
#             'cross_industry_activity': len(industry_stats) > 1
#         }
        
#         return profile

    
#     def _get_cached_category(self, item_name: str) -> str:
#         """Категоризація з кешуванням"""
#         if item_name in self.category_cache:
#             return self.category_cache[item_name]
        
#         categories = self.categories_manager.categorize_item(item_name)
#         category = categories[0][0] if categories else 'unknown'
        
#         # Кешуємо якщо не переповнений
#         if len(self.category_cache) < 10000:
#             self.category_cache[item_name] = category
        
#         return category
    
#     def _save_profiles_batch(self, profiles_batch: Dict[str, SupplierProfile]):
#         """Збереження батчу профілів"""
#         # Додаємо до основного профайлера
#         for edrpou, profile in profiles_batch.items():
#             self.profiler.profiles[edrpou] = profile
        
#         logger.info(f"💾 Збережено {len(profiles_batch)} профілів")
    
#     def _get_memory_usage(self) -> int:        
#         """Отримання використання пам'яті в MB"""
#         import psutil
#         process = psutil.Process()
#         return process.memory_info().rss // 1024 // 1024

# if __name__ == "__main__":
#     import psutil
    
#     # Перевірка пам'яті
#     memory = psutil.virtual_memory()
#     print(f"💾 Доступна пам'ять: {memory.available / (1024**3):.1f} GB")
    
#     if memory.available < 4 * (1024**3):  # менше 4GB
#         print("⚠️ Мало пам'яті! Використовуємо економний режим")
#         batch_size = 5000
#     else:
#         batch_size = 20000
    
#     # Ініціалізація
#     system = TenderAnalysisSystem(
#         categories_file="categories.jsonl",
#         qdrant_host="localhost", 
#         qdrant_port=6333
#     )
#     system.initialize_system()
    
#     # Швидкий білдер
#     builder = FastProfileBuilder(system)
    
#     # ШВИДКИЙ ЗАПУСК
#     results = builder.build_all_profiles_optimized(
#         batch_size=batch_size,
#         save_every=10000
#     )


# # if __name__ == "__main__":
# #     # Опції для різних режимів
# #     import argparse
    
# #     parser = argparse.ArgumentParser(description="Побудова профілів постачальників")
# #     parser.add_argument('--batch-size', type=int, default=10000, help='Розмір батчу')
# #     parser.add_argument('--save-every', type=int, default=5000, help='Зберігати кожні N профілів')
# #     parser.add_argument('--fast', action='store_true', help='Швидкий режим (менше деталей)')
# #     parser.add_argument('--test', action='store_true', help='Тестовий режим (100 профілів)')
    
# #     args = parser.parse_args()
    
# #     # Ініціалізація системи
# #     logger.info("🚀 Ініціалізація системи...")
# #     system = TenderAnalysisSystem(
# #         categories_file="categories.jsonl",
# #         qdrant_host="localhost",
# #         qdrant_port=6333
# #     )
    
# #     if not system.initialize_system():
# #         logger.error("❌ Помилка ініціалізації системи")
# #         exit(1)
    
# #     # Створення оптимізованого побудовника
# #     builder = FastProfileBuilder(system)
    
# #     # Тестовий режим
# #     if args.test:
# #         logger.info("🧪 ТЕСТОВИЙ РЕЖИМ: створення 100 профілів")
# #         # Обмежуємо кількість
# #         all_edrpou = builder._get_all_suppliers_fast()[:100]
# #         builder.all_edrpou = all_edrpou
    
# #     # Запуск побудови
# #     results = builder.build_all_profiles_optimized(
# #         batch_size=args.batch_size,
# #         save_every=args.save_every
# #     )
    
# #     logger.info("🎉 Готово!")