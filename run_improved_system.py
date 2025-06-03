# run_improved_system.py
import json
import sys
from pathlib import Path
from tender_analysis_system import TenderAnalysisSystem
from api_interface import TenderPredictionAPI


def main():
    # 1. Ініціалізація системи
    print("🚀 Запуск покращеної системи аналізу тендерів...")
    
    system = TenderAnalysisSystem(
        categories_file="categories.jsonl"
    )
    
    if not system.initialize_system():
        print("❌ Помилка ініціалізації системи")
        return
    
    # 2. Завантаження даних
    data_file = "your_tender_data.jsonl"  # Вкажіть ваш файл
    
    if not Path(data_file).exists():
        print(f"❌ Файл {data_file} не знайдено")
        return
    
    # Для великих файлів використовуйте:
    # results = system.process_large_dataset(data_file)
    
    # Для звичайних файлів:
    print("📥 Завантаження даних...")
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
            if i >= 10000:  # Обмеження для тесту
                break
    
    # 3. Обробка та тренування
    print("📊 Обробка даних...")
    system.load_and_process_data(data)
    
    print("🎯 Тренування моделі...")
    training_results = system.train_prediction_model()
    print(f"✅ Модель натренована. AUC: {training_results['performance_metrics']['auc_score']:.4f}")
    
    # 4. Створення API
    api = TenderPredictionAPI(system)
    
    # 5. Тестовий прогноз
    print("\n🔮 Тестування прогнозування...")
    test_tender = data[0]  # Беремо перший тендер для тесту
    result = api.predict_single_tender(test_tender)
    
    print(f"\nРезультат прогнозу:")
    print(f"- Ймовірність перемоги: {result['prediction']['win_probability']:.3f}")
    print(f"- Рівень конфіденції: {result['prediction']['confidence']}")
    print(f"- Очікувана конкуренція: {result['competition']['level']}")
    
    # 6. Збереження системи
    print("\n💾 Збереження системи...")
    system.save_system("tender_system_improved.pkl")
    print("✅ Система збережена успішно!")
    
    # 7. Генерація звіту моніторингу
    print("\n📈 Звіт моніторингу:")
    report = system.predictor.monitor.generate_monitoring_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()