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
        # Крок 1: Увімкнення HNSW індексу
        print("\n📈 Крок 1: Увімкнення HNSW індексу...")
        client.update_collection(
            collection_name=collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=16,                          # Увімкнути граф
                ef_construct=200,              # Якісна побудова
                full_scan_threshold=10000,     # Поріг для повного сканування
                max_indexing_threads=4,        # Більше потоків
                on_disk=True,                  # Зберігати на диску
                payload_m=16
            )
        )
        print("✅ HNSW індекс увімкнено")
        
        # Крок 2: Налаштування оптимізаторів
        print("\n⚙️ Крок 2: Оптимізація параметрів...")
        client.update_collection(
            collection_name=collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=4,      # Більше сегментів
                max_segment_size=500000,       # Середні сегменти
                memmap_threshold=20000,        # Стандартний поріг
                indexing_threshold=50000,      # Поріг індексації
                flush_interval_sec=30,         # Частіше скидання
                max_optimization_threads=2    # Потоки оптимізації
            )
        )
        print("✅ Оптимізатори налаштовано")
        
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
        
        # Крок 4: Створення додаткових індексів
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
                    wait=False
                )
                print(f"  ✅ Індекс для {field_name}")
            except Exception as e:
                print(f"  ⚠️ Індекс для {field_name}: {e}")
        
        # Крок 5: Моніторинг прогресу індексації
        print(f"\n⏳ Крок 5: Моніторинг індексації...")
        print("   (Це може зайняти 30-60 хвилин)")
        
        last_indexed = 0
        stable_count = 0
        
        while True:
            time.sleep(30)  # Перевірка кожні 30 секунд
            
            try:
                info = client.get_collection(collection_name)
                
                # Отримуємо інформацію про індексацію
                if hasattr(info, 'points_count') and hasattr(info, 'indexed_vectors_count'):
                    total = info.points_count
                    indexed = getattr(info, 'indexed_vectors_count', total)
                else:
                    # Fallback для старіших версій
                    total = info.points_count
                    indexed = total  # Припускаємо що все проіндексовано
                
                progress = (indexed / total * 100) if total > 0 else 100
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                
                print(f"   📈 Прогрес: {indexed:,}/{total:,} ({progress:.1f}%) | Час: {elapsed:.1f} хв")
                
                # Перевірка завершення
                if indexed == total:
                    if indexed == last_indexed:
                        stable_count += 1
                        if stable_count >= 3:  # Стабільно 1.5 хвилини
                            break
                    else:
                        stable_count = 0
                
                last_indexed = indexed
                
                # Тайм-аут (2 години)
                if elapsed > 120:
                    print("⚠️ Тайм-аут досягнуто, але індексація може продовжуватися")
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
        print(f"⏱️ Загальний час: {total_time:.1f} хвилин")
        print(f"🔍 Час пошуку: {search_time:.1f} мс")
        print(f"📈 Знайдено результатів: {len(results)}")
        print(f"✅ Векторна база готова до використання!")
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
        
        # Тест швидкості пошуку
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