# test_improvements.py
import json
from pathlib import Path
from tender_analysis_system import TenderAnalysisSystem


def test_improved_system():
    """Тестування покращеної системи"""
    
    # 1. Ініціалізація
    system = TenderAnalysisSystem(
        categories_file="categories.jsonl"
    )
    
    system.initialize_system()
    
    # 2. Завантаження тестових даних
    test_data = []
    with open("your_data.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
            if len(test_data) >= 1000:  # Обмежуємо для тесту
                break
    
    # 3. Обробка даних
    print("📊 Обробка даних...")
    results = system.load_and_process_data(test_data)
    print(f"✅ Оброблено: {results}")
    
    # 4. Тренування моделі
    print("\n🎯 Тренування моделі...")
    training_results = system.train_prediction_model()
    print(f"✅ AUC Score: {training_results['performance_metrics']['auc_score']:.4f}")
    
    # 5. Тест прогнозування
    print("\n🔮 Тестування прогнозування...")
    test_items = test_data[:10]  # Беремо 10 записів для тесту
    predictions = system.predict_tender_outcomes(test_items)
    
    for tender_id, pred in predictions['predictions'].items():
        print(f"Тендер {tender_id}: {pred['probability']:.3f} ({pred['confidence']})")
    
    # 6. Тест моніторингу
    print("\n📈 Генерація звіту моніторингу...")
    monitor_report = system.predictor.monitor.generate_monitoring_report()
    print(f"Всього прогнозів: {monitor_report['total_predictions']}")
    
    # 7. Збереження системи
    print("\n💾 Збереження системи...")
    system.save_system("tender_system_improved.pkl")
    print("✅ Система збережена")


if __name__ == "__main__":
    test_improved_system()