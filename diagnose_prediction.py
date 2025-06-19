# diagnose_prediction.py
import json
from tender_analysis_system import TenderAnalysisSystem

# Завантажуємо систему
system = TenderAnalysisSystem()
system.initialize_system()
system.load_system("tender_system_trained.pkl")

# Завантажуємо профілі
with open('supplier_profiles_with_clusters.json', 'r', encoding='utf-8') as f:
    profiles = json.load(f)

# Тестовий тендер
test_tender = {
    "EDRPOU": "36074695",
    "F_ITEMNAME": "SIEMENS 3RT2016-2BB41",
    "F_TENDERNAME": "Закупівля електротехнічної продукції з орієнтовною потребою на 2025р",
    "F_INDUSTRYNAME": "Електротовари, кабельнопровідникова продукція"
}

# Отримуємо профіл
profile = profiles.get("36074695")
print("📊 ПРОФІЛЬ ПОСТАЧАЛЬНИКА:")
print(f"Назва: {profile['name']}")
print(f"Win rate: {profile['metrics']['win_rate']:.1%}")
print(f"Досвід в категорії: {profile['categories'].get('Електротовари, кабельнопровідникова продукція', {}).get('win_rate', 0):.1%}")
print(f"Бренди: {profile.get('brand_expertise', [])}")

# Витягуємо всі ознаки
features = system.feature_extractor.extract_features(test_tender, profile)

print("\n🔍 ВИТЯГНУТІ ОЗНАКИ:")
# Ключові ознаки постачальника
print(f"\nОзнаки постачальника:")
print(f"  supplier_win_rate: {features.get('supplier_win_rate', 0):.3f}")
print(f"  supplier_experience: {features.get('supplier_experience', 0)}")
print(f"  supplier_category_win_rate: {features.get('supplier_category_win_rate', 0):.3f}")
print(f"  supplier_category_experience: {features.get('supplier_category_experience', 0)}")

# Бренди
print(f"\nБрендові ознаки:")
print(f"  has_brand: {features.get('has_brand', 0)}")
print(f"  brand_count: {features.get('brand_count', 0)}")

# Конкуренція
print(f"\nКонкурентні ознаки:")
print(f"  competition_intensity: {features.get('competition_intensity', 0):.3f}")
print(f"  market_concentration: {features.get('market_concentration', 0):.3f}")

# Перевірка категорії
print(f"\nКатегорійні ознаки:")
print(f"  primary_category: {features.get('primary_category', '')}")
print(f"  category_confidence: {features.get('category_confidence', 0):.3f}")

# Всі ознаки
print(f"\n📋 ВСІ ОЗНАКИ ({len(features)}):")
for key, value in sorted(features.items()):
    if isinstance(value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")

# Прогнозування
predictions = system.predict_tender_outcomes([test_tender])
print("\n📈 ПРОГНОЗ: ")
# Якщо predictions — це рядок (наприклад, JSON), розпарсити його у список словників
print(predictions)
