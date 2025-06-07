# test_training.py
from tender_analysis_system import TenderAnalysisSystem
import json

# 1. Ініціалізація системи
print("🚀 Ініціалізація системи...")
system = TenderAnalysisSystem()
system.initialize_system()

# 2. Завантаження тестових даних
print("\n📥 Завантаження даних...")
test_data = []
with open("out_10_nodup.jsonl", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10000:  # Беремо перші 10k для тесту
            break
        test_data.append(json.loads(line))

print(f"✅ Завантажено {len(test_data)} записів")

# 3. Обробка даних і створення профілів
print("\n📊 Обробка даних...")
results = system.load_and_process_data(test_data)
print(f"✅ Створено профілів: {results['new_suppliers']}")

# 4. Тренування моделі
print("\n🎯 Тренування моделі...")
try:
    # Підготовка даних для тренування
    system.predictor.prepare_training_data_from_history(
        test_data,
        system.supplier_profiler.get_all_profiles()
    )
    
    # Тренування
    training_results = system.train_prediction_model()
    print(f"✅ AUC Score: {training_results['performance_metrics']['auc_score']:.4f}")
    
    # Збереження системи
    print("\n💾 Збереження системи...")
    system.save_system("tender_system_trained.pkl")
    print("✅ Система збережена!")
    
except Exception as e:
    print(f"❌ Помилка: {e}")
    import traceback
    traceback.print_exc()