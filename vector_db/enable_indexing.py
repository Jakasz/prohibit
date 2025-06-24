#!/usr/bin/env python3
"""
Виправлена версія скрипта для увімкнення індексації з більшим таймаутом
"""

import time
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
import httpx


def enable_indexing_with_timeout(collection_name: str = "tender_vectors", 
                                host: str = "localhost", 
                                port: int = 6333):
    """Увімкнення індексації з великим таймаутом"""
    
    print("="*60)
    print("🔧 УВІМКНЕННЯ ІНДЕКСАЦІЇ (з виправленим таймаутом)")
    print("="*60)
    
    # Створення клієнта з ВЕЛИКИМ таймаутом
    try:
        client = QdrantClient(
            host=host, 
            port=port,
            timeout=300,  # 5 хвилин базовий таймаут
            # Додаткові налаштування для великих операцій
            grpc_options={
                "keepalive_time_ms": 10000,
                "keepalive_timeout_ms": 5000,
                "keepalive_permit_without_calls": True,
                "http2_max_pings_without_data": 0,
            } if port == 6334 else None
        )
        print(f"✅ Підключено до Qdrant: {host}:{port}")
    except Exception as e:
        print(f"❌ Помилка підключення: {e}")
        return False
    
    # Перевірка колекції
    try:
        collection_info = client.get_collection(collection_name)
        points_count = collection_info.points_count
        current_status = collection_info.status
        
        print(f"\n📊 Колекція '{collection_name}':")
        print(f"   • Записів: {points_count:,}")
        print(f"   • Статус: {current_status}")
        
        # Перевірка чи вже йде індексація
        if current_status == "yellow":
            print("\n⚠️  Індексація вже виконується!")
            print("   Qdrant продовжує індексувати у фоновому режимі.")
            
            response = input("\nПродовжити моніторинг? (y/n): ")
            if response.lower() == 'y':
                monitor_indexing(client, collection_name)
            return True
            
    except Exception as e:
        print(f"❌ Колекція '{collection_name}' не знайдена: {e}")
        return False
    
    # Запит підтвердження
    print(f"\n🤔 Увімкнути індексацію для {points_count:,} записів?")
    print("⏱️  Це може зайняти 1-3 години для 30GB")
    response = input("Продовжити? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ Операція скасована")
        return False
    
    start_time = datetime.now()
    print(f"\n🚀 Початок: {start_time.strftime('%H:%M:%S')}")
    
    # Поетапне увімкнення індексації
    try:
        # Крок 1: М'які налаштування оптимізації
        print("\n⚙️  Крок 1: Налаштування оптимізаторів...")
        try:
            client.update_collection(
                collection_name=collection_name,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=50000,  # Помірний поріг
                    flush_interval_sec=60,     # Частіше скидання
                    max_optimization_threads=4  # Більше потоків
                ),
                timeout=60  # 1 хвилина на цю операцію
            )
            print("✅ Оптимізатори налаштовано")
        except Exception as e:
            print(f"⚠️  Таймаут при оновленні оптимізаторів (це нормально): {e}")
        
        # Даємо час на застосування
        time.sleep(5)
        
        # Крок 2: Увімкнення HNSW поступово
        print("\n📈 Крок 2: Увімкнення HNSW індексу...")
        try:
            # Спочатку базові параметри
            client.update_collection(
                collection_name=collection_name,
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,  # Менше значення для швидшої побудови
                    full_scan_threshold=20000,
                    on_disk=True
                ),
                timeout=60
            )
            print("✅ HNSW увімкнено з базовими параметрами")
        except Exception as e:
            print(f"⚠️  Таймаут при увімкненні HNSW (це нормально): {e}")
        
        # Крок 3: Форсування індексації
        print("\n🔨 Крок 3: Запуск індексації...")
        try:
            client.update_collection(
                collection_name=collection_name,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=1000  # Низький поріг для старту
                ),
                timeout=30
            )
            print("✅ Індексація запущена")
        except Exception as e:
            print(f"⚠️  Таймаут при запуску (це нормально): {e}")
        
        print("\n✅ Всі команди відправлено!")
        print("📋 Qdrant тепер виконує індексацію у фоновому режимі")
        
        # Пропозиція моніторингу
        response = input("\n📊 Запустити моніторинг прогресу? (y/n): ")
        if response.lower() == 'y':
            monitor_indexing(client, collection_name)
        else:
            print("\n💡 Підказка: Ви можете перевірити статус пізніше командою:")
            print("   python enable_indexing.py status")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        return False


def monitor_indexing(client: QdrantClient, collection_name: str):
    """Моніторинг прогресу індексації"""
    print("\n⏳ Моніторинг індексації...")
    print("(Натисніть Ctrl+C для виходу)")
    
    last_status = None
    stable_green_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            try:
                info = client.get_collection(collection_name)
                current_status = info.status
                
                # Вивід статусу якщо змінився
                if current_status != last_status:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Статус: {current_status}")
                    last_status = current_status
                    
                    if current_status == "green":
                        stable_green_count += 1
                    else:
                        stable_green_count = 0
                
                # Перевірка завершення
                if stable_green_count >= 3:
                    print("\n✅ Індексація завершена!")
                    
                    # Фінальний тест
                    test_search_performance(client, collection_name)
                    
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    print(f"\n⏱️  Загальний час: {elapsed:.1f} хвилин")
                    break
                
                # Прогрес кожні 5 хвилин
                elapsed = (datetime.now() - start_time).total_seconds()
                if int(elapsed) % 300 == 0:  # кожні 5 хвилин
                    print(f"   ⏱️  Пройшло: {elapsed/60:.1f} хв")
                
                time.sleep(30)  # Перевірка кожні 30 секунд
                
            except Exception as e:
                print(f"⚠️  Помилка моніторингу: {e}")
                time.sleep(60)
                
    except KeyboardInterrupt:
        print("\n⚠️  Моніторинг зупинено")
        print("💡 Індексація продовжується у фоні")


def test_search_performance(client: QdrantClient, collection_name: str):
    """Тест швидкості пошуку"""
    print("\n🧪 Тестування пошуку...")
    
    test_vector = [0.1] * 768
    
    # Кілька тестів
    times = []
    for i in range(5):
        start = time.time()
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=test_vector,
                limit=10,
                timeout=10
            )
            search_time = (time.time() - start) * 1000
            times.append(search_time)
            print(f"   Тест {i+1}: {search_time:.1f} мс ({len(results)} результатів)")
        except Exception as e:
            print(f"   Тест {i+1}: Помилка - {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n📊 Середній час пошуку: {avg_time:.1f} мс")
        
        if avg_time < 50:
            print("✅ Відмінна швидкість!")
        elif avg_time < 100:
            print("✅ Хороша швидкість")
        elif avg_time < 500:
            print("⚠️  Прийнятна швидкість")
        else:
            print("❌ Повільний пошук - індексація може бути не завершена")


def check_status(collection_name: str = "tender_vectors"):
    """Перевірка статусу індексації"""
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=30)
        info = client.get_collection(collection_name)
        
        print(f"\n📊 Статус колекції '{collection_name}':")
        print(f"   • Записів: {info.points_count:,}")
        print(f"   • Статус: {info.status}")
        
        # Тест швидкості
        test_search_performance(client, collection_name)
        
    except Exception as e:
        print(f"❌ Помилка: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_status()
    else:
        enable_indexing_with_timeout()