from collections import defaultdict
import json
import logging
import os
from typing import Dict, List

from tqdm import tqdm
from supplier_profiler import SupplierMetrics, SupplierProfile
from tender_analysis_system import TenderAnalysisSystem

import pickle

class ProfileBuilder:    
    def __init__(self, system: TenderAnalysisSystem):
        self.system = system
        self.vector_db = system.vector_db
        self.profiler = system.supplier_profiler
        self.logger = logging.getLogger(__name__)

    def build_and_cache(self):
        self.logger.info("🚨 РЕЖИМ ПІДГОТОВКИ ТА КЕШУВАННЯ ДАНИХ")
        cache_file = "files/all_data_cache.pkl"
        supplier_data = defaultdict(list)
        if os.path.exists(cache_file):
            self.logger.info("📦 Знайдено кеш даних, завантаження пропущено.")
            return
        # 1. Завантаження через scroll
        self.logger.info("⚡ Початок завантаження...")
        collection_info = self.vector_db.client.get_collection(
            collection_name=self.vector_db.collection_name
        )
        total_points = collection_info.points_count
        self.logger.info(f"📊 Всього записів в БД: {total_points:,}")
        pbar = tqdm(total=total_points, desc="Завантаження даних", unit="записів")
        offset = None
        total_loaded = 0
        batch_size = 40000
        while True:
            try:
                records, next_offset = self.vector_db.client.scroll(
                    collection_name=self.vector_db.collection_name,
                    offset=offset,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=False
                )
                if not records:
                    break
                for record in records:
                    if record.payload:
                        edrpou = (record.payload.get('edrpou') or 
                                  record.payload.get('EDRPOU') or 
                                  '')
                        if edrpou:
                            supplier_data[edrpou].append(record.payload)
                            total_loaded += 1
                            pbar.update(1)
                        else:
                            self.logger.debug(f"Запис без ЄДРПОУ: {record.payload}")
                if total_loaded % 50000 == 0:
                    self.logger.info(f"   Завантажено {total_loaded:,} записів, унікальних ЄДРПОУ: {len(supplier_data):,}")
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
        self.logger.info(f"✅ Кеш збережено у {cache_file}")
if __name__ == "__main__":
    system = TenderAnalysisSystem(
        categories_file="data/categories.jsonl",
        qdrant_host="localhost",
        qdrant_port=6333
    )
    builder = ProfileBuilder(system)
    builder.build_and_cache()