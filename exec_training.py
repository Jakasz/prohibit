# 1. Ініціалізація системи
from pathlib import Path
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
profiles_file = "supplier_profiles_COMPLETE.json"
if Path(profiles_file).exists():
    system.supplier_profiler.load_profiles(profiles_file)
    print(f"✅ Завантажено {len(system.supplier_profiler.profiles)} профілів")
else:
    print("⚠️ Профілі не знайдено. Навчання буде без них (гірша якість)")

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

# 6. Тестове передбачення
test_tender = {
    "EDRPOU": "12345678",
    "F_ITEMNAME": "Фільтр паливний для трактора John Deere",
    "F_TENDERNUMBER": "UA-2024-01-01-000001",
    "F_INDUSTRYNAME": "Сільське господарство",
    "CPV": "16810000"
}

predictions = system.predict_tender_outcomes([test_tender])
print(f"\n🔮 Тестове передбачення:")
print(f"Ймовірність перемоги: {predictions['predictions'].get(test_tender['F_TENDERNUMBER'], {}).get('probability', 0):.2%}")