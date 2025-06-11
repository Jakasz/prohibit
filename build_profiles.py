from tender_analysis_system import TenderAnalysisSystem

if __name__ == "__main__":
    # 1. Ініціалізуємо систему (підключення до Qdrant, категорії, профайлер)
    system = TenderAnalysisSystem(
        categories_file="categories.jsonl",  # або ваш шлях до категорій
        qdrant_host="localhost",
        qdrant_port=6333
    )
    system.initialize_system()

    # 2. Створюємо профілі постачальників на основі даних з векторної бази
    profiler = system.supplier_profiler
    results = profiler.build_profiles(update_mode=True)  # або False, якщо не оновлювати існуючі

    # 3. Зберігаємо профілі у файл
    profiler.save_profiles("supplier_profiles.json")

    print(f"✅ Профілювання завершено. Створено/оновлено профілів: {results}")