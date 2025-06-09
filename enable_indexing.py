#!/usr/bin/env python3
"""
Скрипт для увімкнення індексації після швидкого завантаження
"""

import time
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models


def enable_indexing(collection_name: str = "tender_vectors", 
                   host: str = "localhost", 
                   port: int = 6333):
    """Увімкнення індексації для колекції"""
    
    print("="*60)
    print("🔧 УВІМКНЕННЯ ІНДЕКСАЦІЇ ВЕКТОРНОЇ БАЗИ")
    print("="*60)
    
    # Підключення
    try:
        client = QdrantClient(host=host, port=port)
        print(f"✅ Підключено до Qdrant: {host}:{port}")
    except Exception as e:
        print(f"❌ Помилка підключення: {e}")
        return False
    
    # Перевірка колекції
    try:
        collection_info = client.get_collection(collection_name)
        points_count = collection_info.points_count
        print(f"📊 Колекція '{collection_name}': {points_count:,} записів")
        
        if points_count == 0:
            print("⚠️ Колекція порожня, індексацію не потрібно")
            return False
            
    except Exception as e:
        print(f"❌ Колекція '{collection_name}' не знайдена: {e}")
        return False
    
    # Запит підтвердження
    print(f"\n🤔 Увімкнути індексацію для {points_count:,} записів?")
    print("⏱️ Це може зайняти 30-60 хвилин")
    response = input("Продовжити? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Операція скасована")
        return False
    
    start_time = datetime.now()
    print(f"\n🚀 Початок індексації: {start_time.strftime('%H:%M:%S')}")
    
    try:
        # Крок 1: Налаштування оптимізаторів для швидшої індексації
        print("\n⚙️ Крок 1: Оптимізація параметрів для індексації...")
        client.update_collection(
            collection_name=collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=4,      # Більше сегментів для паралельності
                max_segment_size=500000,       # Середні сегменти
                memmap_threshold=20000,        # Стандартний поріг
                indexing_threshold=20000,      # УВІМКНУТИ автоматичну індексацію
                flush_interval_sec=30,         # Частіше скидання
                max_optimization_threads=4     # Більше потоків для оптимізації
            )
        )
        print("✅ Оптимізатори налаштовано для індексації")
        
        # Крок 2: Увімкнення HNSW індексу
        print("\n📈 Крок 2: Увімкнення HNSW індексу...")
        client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=16,                          # Стандартне значення для графу
                ef_construct=200,              # Якісна побудова
                full_scan_threshold=10000,     # Поріг для повного сканування
                max_indexing_threads=0,        # 0 = всі доступні ядра
                on_disk=True,                  # Зберігати на диску
                payload_m=16                   # Індексація payload
            )
        )
        print("✅ HNSW індекс увімкнено")
        
        # Крок 3: Увімкнення квантизації (опціонально)
        print("\n🗜️ Крок 3: Увімкнення квантизації...")
        try:
            client.update_collection(
                collection_name=collection_name,
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=False
                    )
                )
            )
            print("✅ Квантизація увімкнена")
        except Exception as e:
            print(f"⚠️ Квантизацію не вдалося увімкнути: {e}")
        
        # Крок 4: Створення додаткових індексів для payload
        print("\n📋 Крок 4: Створення додаткових індексів...")
        additional_indexes = [
            ("industry", models.PayloadSchemaType.KEYWORD),
            ("primary_category", models.PayloadSchemaType.KEYWORD),
            ("cpv", models.PayloadSchemaType.INTEGER),
            ("budget", models.PayloadSchemaType.FLOAT),
            ("date_end", models.PayloadSchemaType.KEYWORD),
            ("owner_name", models.PayloadSchemaType.KEYWORD)
        ]
        
        for field_name, field_type in additional_indexes:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                    wait=False  # Асинхронне створення
                )
                print(f"  ✅ Індекс для {field_name}")
            except Exception as e:
                print(f"  ⚠️ Індекс для {field_name}: {e}")
        
        # Крок 5: Форсування індексації
        print(f"\n🔨 Крок 5: Форсування індексації...")
        print("   (Це може зайняти 30-60 хвилин)")
        
        # Оновлюємо колекцію для запуску індексації
        client.update_collection(
            collection_name=collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=1000  # Низький поріг для швидкого початку
            )
        )
        
        # Моніторинг прогресу
        print("\n⏳ Моніторинг індексації...")
        last_status = None
        stable_count = 0
        check_interval = 30  # секунди
        
        while True:
            time.sleep(check_interval)
            
            try:
                info = client.get_collection(collection_name)
                
                # Статус колекції
                current_status = info.status
                
                if current_status != last_status:
                    print(f"   📊 Статус: {current_status}")
                    last_status = current_status
                    stable_count = 0
                else:
                    stable_count += 1
                
                # Перевірка завершення
                if current_status == "green" and stable_count >= 3:
                    print("   ✅ Індексація завершена!")
                    break
                
                # Час
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                print(f"   ⏱️ Пройшло: {elapsed:.1f} хв")
                
                # Тайм-аут (2 години)
                if elapsed > 120:
                    print("⚠️ Тайм-аут досягнуто, індексація може продовжуватися в фоні")
                    break
                    
            except Exception as e:
                print(f"   ⚠️ Помилка моніторингу: {e}")
                break
        
        # Фінальна перевірка
        print(f"\n🔍 Фінальна перевірка...")
        final_info = client.get_collection(collection_name)
        
        # Тестовий пошук
        print(f"🧪 Тест пошуку...")
        test_vector = [0.1] * 768
        search_start = time.time()
        
        results = client.search(
            collection_name=collection_name,
            query_vector=test_vector,
            limit=5
        )
        
        search_time = (time.time() - search_start) * 1000
        
        total_time = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "="*60)
        print("🎉 ІНДЕКСАЦІЯ ЗАВЕРШЕНА!")
        print("="*60)
        print(f"📊 Записів у колекції: {final_info.points_count:,}")
        print(f"📊 Статус колекції: {final_info.status}")
        print(f"⏱️ Загальний час: {total_time:.1f} хвилин")
        print(f"🔍 Час пошуку: {search_time:.1f} мс")
        print(f"📈 Знайдено результатів: {len(results)}")
        
        if search_time < 100:
            print(f"✅ Векторна база працює ВІДМІННО!")
        elif search_time < 500:
            print(f"✅ Векторна база працює добре")
        else:
            print(f"⚠️ Векторна база працює повільно")
        
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Помилка індексації: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_indexing_status(collection_name: str = "tender_vectors"):
    """Перевірка статусу індексації"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        info = client.get_collection(collection_name)
        
        print(f"\n📊 Статус колекції '{collection_name}':")
        print(f"   • Записів: {info.points_count:,}")
        print(f"   • Статус: {info.status}")
        
        # Перевірка конфігурації
        print(f"\n⚙️ Конфігурація:")
        if hasattr(info.config, 'optimizer_config'):
            opt_config = info.config.optimizer_config
            print(f"   • Поріг індексації: {opt_config.indexing_threshold}")
        
        if hasattr(info.config, 'hnsw_config'):
            hnsw_config = info.config.hnsw_config
            print(f"   • HNSW m: {hnsw_config.m}")
            print(f"   • HNSW ef_construct: {hnsw_config.ef_construct}")
        
        # Тест швидкості пошуку
        print(f"\n🧪 Тест пошуку...")
        test_vector = [0.1] * 768
        start_time = time.time()
        
        results = client.search(
            collection_name=collection_name,
            query_vector=test_vector,
            limit=5
        )
        
        search_time = (time.time() - start_time) * 1000
        print(f"   • Швидкість пошуку: {search_time:.1f} мс")
        print(f"   • Результатів знайдено: {len(results)}")
        
        if search_time < 100:
            print("   ✅ Індексація працює відмінно")
        elif search_time < 500:
            print("   ⚠️ Індексація працює, але може бути не завершена")
        else:
            print("   ❌ Індексація не працює або не завершена")
            
    except Exception as e:
        print(f"❌ Помилка перевірки: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_indexing_status()
    else:
        success = enable_indexing()
        if not success:
            sys.exit(1)