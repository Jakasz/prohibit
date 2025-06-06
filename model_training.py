# train_system.py
from tender_analysis_system import TenderAnalysisSystem
import json

# Ініціалізація
system = TenderAnalysisSystem()
system.initialize_system()

# Завантаження історичних даних
data = []
with open("your_historical_data.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Обробка даних
system.load_and_process_data(data)

# Тренування моделі
system.train_prediction_model()

# Збереження системи
system.save_system("tender_system.pkl")
print("✅ Система готова!")