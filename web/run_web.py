#!/usr/bin/env python3
"""
Скрипт для запуску веб-інтерфейсу системи прогнозування тендерів
"""

import os
import sys
import logging
from pathlib import Path

# Перевірка необхідних файлів
required_files = [
    'tender_analysis_system.py',
    'category_manager.py',
    'supplier_profiler.py',
    'competition_analyzer.py',
    'prediction_engine.py',
    'feature_extractor.py',
    'vector_database.py'
]

missing_files = []
for file in required_files:
    if not Path(file).exists():
        missing_files.append(file)

if missing_files:
    print("❌ Відсутні необхідні файли:")
    for file in missing_files:
        print(f"   - {file}")
    print("\nПереконайтеся, що всі файли проекту знаходяться в поточній директорії")
    sys.exit(1)

# Перевірка наявності навченої моделі
model_files = [
    'tender_system_trained.pkl'
]

model_found = any(Path(f).exists() for f in model_files)
if not model_found:
    print("⚠️ УВАГА: Не знайдено жодної навченої моделі!")
    print("Модель повинна бути навчена перед використанням.")
    print("\nЗапустіть навчання командою:")
    print("   python exec_training.py")
    response = input("\nПродовжити без моделі? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

# Перевірка профілів
profile_files = [
    'supplier_profiles_with_clusters.json'
]

profiles_found = any(Path(f).exists() for f in profile_files)
if not profiles_found:
    print("⚠️ УВАГА: Не знайдено профілів постачальників!")
    print("Рекомендується створити профілі для кращої якості прогнозів.")

# Створення директорії для шаблонів якщо не існує
if not Path('templates').exists():
    Path('templates').mkdir()
    print("✅ Створено директорію templates/")

print("\n" + "="*60)
print("🚀 ЗАПУСК ВЕБ-ІНТЕРФЕЙСУ СИСТЕМИ ПРОГНОЗУВАННЯ")
print("="*60)
print("\n📋 Інструкції:")
print("1. Відкрийте браузер за адресою: http://localhost:5000")
print("2. Система автоматично ініціалізується при першому запуску")
print("3. Для зупинки натисніть Ctrl+C")
print("\n" + "="*60 + "\n")

# Запуск Flask додатку
try:
    from web_app import app
    app.run(debug=False, host='0.0.0.0', port=5000)
except KeyboardInterrupt:
    print("\n\n✅ Веб-сервер зупинено")
except Exception as e:
    print(f"\n❌ Помилка запуску: {e}")
    print("\nПереконайтеся, що встановлені всі залежності:")
    print("   pip install -r requirements_web.txt")