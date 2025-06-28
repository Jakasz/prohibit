# 1. Ініціалізація системи
from pathlib import Path

import pandas as pd
from tender_analysis_system import TenderAnalysisSystem



system = TenderAnalysisSystem(
    categories_file="categories.jsonl",
    qdrant_host="localhost",
    qdrant_port=6333
)
system.initialize_system()

# 2. Перевірка векторної бази
db_size = system.vector_db.get_collection_size()
print(f"📊 Записів у векторній базі: {db_size:,}")

if db_size < 1000:
    print("❌ Недостатньо даних для навчання!")

exit(1)

# 3. ОПЦІОНАЛЬНО: Завантаження профілів постачальників (покращує якість)
profiles_files = [
    "supplier_profiles_with_clusters.json"
]
profiles_loaded = False

for profiles_file in profiles_files:
    if Path(profiles_file).exists():
        system.supplier_profiler.load_profiles(profiles_file)
        print(f"✅ Завантажено {len(system.supplier_profiler.profiles)} профілів з {profiles_file}")
        profiles_loaded = True
        break

# if not profiles_loaded:
#     print("⚠️ Профілі не знайдено. Створюємо нові...")
    
#     # Імпортуємо та запускаємо створення
#     from update_with_clusters import ProfileBuilderWithClusters
    
#     builder = ProfileBuilderWithClusters(system.vector_db)
#     profiles = builder.build_profiles_from_vector_db()
#     builder.save_profiles()
    
#     # Завантажуємо створені профілі
#     system.supplier_profiler.load_profiles("supplier_profiles_with_clusters.json")
#     print(f"✅ Створено та завантажено {len(system.supplier_profiler.profiles)} профілів")


# 4. Навчання моделі НА ДАНИХ З ВЕКТОРНОЇ БАЗИ
try:
    results = system.train_prediction_model(
        validation_split=0.2,  # 20% на валідацію
        cv_folds=5            # 5-fold cross-validation
    )
    
    print("\n✅ Модель успішно навчена!")
    print(f"📊 Використано зразків: {results['training_samples']:,}")
    print(f"📊 Кількість ознак: {results['feature_count']}")
    print(f"📊 Частка перемог: {results['positive_rate']:.1%}")
    print(f"📊 AUC Score: {results['performance_metrics'].get('ensemble', {}).get('test_auc', 0):.4f}")
    
except Exception as e:
    print(f"❌ Помилка навчання: {e}")
    exit(1)

# 5. Збереження навченої системи
system.save_system("tender_system_trained.pkl")
print("💾 Система збережена!")

print("\n🔍 SHAP АНАЛІЗ МОДЕЛІ")
print("="*60)

try:
    # Запускаємо SHAP аналіз
    shap_results = system.predictor.analyze_with_shap(
        sample_size=2000,  # Аналізуємо 2000 прикладів
        save_plots=True    # Зберігаємо графіки
    )
    
    print("\n✅ SHAP аналіз завершено!")
    print("📊 Графіки збережено в папці shap_plots/")
    print("📄 Детальний звіт в shap_analysis_report.json")
    
except Exception as e:
    print(f"⚠️ Не вдалося виконати SHAP аналіз: {e}")

# 7. Тестове пояснення одного прогнозу
print("\n🎯 ПРИКЛАД ПОЯСНЕННЯ КОНКРЕТНОГО ПРОГНОЗУ")
print("="*60)

# Тестові дані у правильному форматі
test_example = {
    "EDRPOU": "12345678",
    "F_ITEMNAME": "Фільтр паливний для трактора John Deere",
    "F_TENDERNAME": "Закупівля запчастин для сільськогосподарської техніки",
    "F_INDUSTRYNAME": "Сільськогосподарські запчастини"
}

try:
    # Опціонально: якщо маємо профіль постачальника
    supplier_profile = None
    if system.supplier_analyzer:
        supplier_profile = system.supplier_analyzer.get_supplier_profile(test_example['EDRPOU'])
    
    # Викликаємо пояснення
    explanation = system.predictor.explain_single_prediction(
        test_example, 
        supplier_profile=supplier_profile,
        show_plot=False  # True якщо хочете графік
    )
    
except Exception as e:
    print(f"Помилка пояснення: {e}")
    import traceback
    traceback.print_exc()

