#!/usr/bin/env python3
"""
Детальний аналіз прогнозу та порівняння моделей
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tender_analysis_system import TenderAnalysisSystem

# Налаштування візуалізації
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_shap_analysis(filepath='shap_analysis_report.json'):
    """Завантаження SHAP звіту"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        print(f"⚠️ Не вдалося завантажити {filepath}")
        return None

def analyze_single_prediction(system, test_data):
    """Детальний аналіз одного прогнозу"""
    print("\n" + "="*80)
    print("🔍 ДЕТАЛЬНИЙ АНАЛІЗ ПРОГНОЗУ")
    print("="*80)
    
    # Базова інформація
    print(f"\n📋 Вхідні дані:")
    print(f"   ЄДРПОУ: {test_data['EDRPOU']}")
    print(f"   Товар: {test_data['F_ITEMNAME']}")
    print(f"   Тендер: {test_data['F_TENDERNAME']}")
    print(f"   Категорія: {test_data['F_INDUSTRYNAME']}")
    
    # Знаходимо профіль постачальника
    supplier_profile = None
    if system.supplier_profiler and hasattr(system.supplier_profiler, 'profiles'):
        supplier_profile = system.supplier_profiler.profiles.get(test_data['EDRPOU'])
        if supplier_profile:
            print(f"\n✅ Знайдено профіль постачальника:")
            print(f"   Назва: {supplier_profile.name}")
            print(f"   Загальний win rate: {supplier_profile.metrics.win_rate:.2%}")
            print(f"   Всього тендерів: {supplier_profile.metrics.total_tenders}")
            print(f"   Виграно тендерів: {supplier_profile.metrics.won_tenders}")
        else:
            print(f"\n⚠️ Профіль постачальника НЕ знайдено!")
    
    # Витягуємо ознаки
    features = system.feature_extractor.extract_features(test_data, supplier_profile)
    
    print(f"\n📊 Витягнуті ознаки (всього {len(features)}):")
    print("-"*80)
    
    # Групуємо ознаки за категоріями
    experience_features = ['has_category_experience', 'experience_type', 'supplier_category_experience', 
                          'supplier_category_win_rate', 'supplier_category_wins']
    supplier_features = ['supplier_win_rate', 'supplier_position_win_rate', 'supplier_experience',
                        'supplier_stability', 'supplier_specialization', 'supplier_recent_win_rate',
                        'supplier_growth_rate', 'supplier_reliability']
    market_features = ['category_avg_suppliers', 'category_win_probability', 'category_market_openness',
                      'category_entry_barrier', 'is_new_supplier', 'supplier_vs_market_avg']
    competition_features = ['competitor_top_avg_win_rate', 'supplier_vs_top_competitors']
    other_features = ['has_brand', 'supplier_category_fit', 'competitive_strength']
    
    # Виводимо по групах
    print("\n🎯 ДОСВІД В КАТЕГОРІЇ:")
    for feat in experience_features:
        if feat in features:
            value = features[feat]
            print(f"   {feat:40}: {value:10.4f}")
    
    print("\n📈 МЕТРИКИ ПОСТАЧАЛЬНИКА:")
    for feat in supplier_features:
        if feat in features:
            value = features[feat]
            print(f"   {feat:40}: {value:10.4f}")
    
    print("\n🏪 РИНКОВІ ПОКАЗНИКИ:")
    for feat in market_features:
        if feat in features:
            value = features[feat]
            print(f"   {feat:40}: {value:10.4f}")
    
    print("\n🥊 КОНКУРЕНЦІЯ:")
    for feat in competition_features:
        if feat in features:
            value = features[feat]
            print(f"   {feat:40}: {value:10.4f}")
    
    print("\n🔧 ІНШІ ОЗНАКИ:")
    for feat in other_features:
        if feat in features:
            value = features[feat]
            print(f"   {feat:40}: {value:10.4f}")
    
    # Створюємо DataFrame для предикції
    X_single = pd.DataFrame([features])
    
    # Заповнюємо відсутні колонки
    for col in system.predictor.feature_names:
        if col not in X_single.columns:
            X_single[col] = 0
    X_single = X_single[system.predictor.feature_names]
    
    # Додаємо interaction features
    X_single = system.feature_extractor.create_interaction_features(X_single)
    
    # Обробка через feature processor
    if hasattr(system.predictor, 'feature_processor'):
        X_processed = system.predictor.feature_processor.transform(X_single)
    else:
        X_processed = X_single
    
    # Масштабування
    X_scaled = system.predictor.scalers['main'].transform(X_processed)
    
    # Прогнози від кожної моделі
    print("\n🎲 ПРОГНОЗИ МОДЕЛЕЙ:")
    print("-"*80)
    
    predictions = {}
    for model_name, model in system.predictor.models.items():
        pred = model.predict_proba(X_scaled)[0][1]
        predictions[model_name] = pred
        print(f"   {model_name:20}: {pred:8.2%}")
    
    # Ансамбль
    ensemble_weights = system.predictor.ensemble_weights
    ensemble_pred = sum(predictions[name] * ensemble_weights.get(name, 1.0) for name in predictions)
    ensemble_pred /= sum(ensemble_weights.values())
    
    print(f"\n   {'ENSEMBLE':20}: {ensemble_pred:8.2%} (фінальний прогноз)")
    print(f"\n   Ваги ансамблю: {ensemble_weights}")
    
    return {
        'features': features,
        'predictions': predictions,
        'ensemble_prediction': ensemble_pred
    }

def compare_feature_importance(shap_report):
    """Порівняння важливості ознак між моделями"""
    if not shap_report:
        print("⚠️ SHAP звіт не доступний")
        return
    
    print("\n" + "="*80)
    print("📊 ПОРІВНЯННЯ ВАЖЛИВОСТІ ОЗНАК")
    print("="*80)
    
    feature_analysis = shap_report.get('feature_analysis', {})
    
    # Збираємо топ-15 найважливіших ознак
    all_features = {}
    for feature, model_data in feature_analysis.items():
        total_importance = sum(data['mean_abs_shap'] for data in model_data.values())
        all_features[feature] = total_importance
    
    # Сортуємо
    top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print("\n🏆 ТОП-15 НАЙВАЖЛИВІШИХ ОЗНАК:")
    print("-"*80)
    print(f"{'Ознака':45} {'Важливість':>12} {'Вплив':>10}")
    print("-"*80)
    
    for feature, importance in top_features:
        # Середній напрямок впливу
        directions = []
        for model_data in feature_analysis[feature].values():
            if model_data['positive_impact_ratio'] > 0.5:
                directions.append('↑')
            else:
                directions.append('↓')
        
        # Найчастіший напрямок
        direction = max(set(directions), key=directions.count) if directions else '?'
        
        print(f"{feature:45} {importance:12.6f} {direction:>10}")
    
    # Візуалізація
    if len(top_features) > 0:
        plt.figure(figsize=(10, 8))
        features, importances = zip(*top_features)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
                  '#BC4749', '#386641', '#F2CC8F', '#81B29A', '#F07167',
                  '#AFD5AA', '#F4A261', '#E76F51', '#264653', '#2A9D8F']
        
        bars = plt.barh(range(len(features)), importances, color=colors[:len(features)])
        plt.yticks(range(len(features)), features)
        plt.xlabel('Сумарна важливість (SHAP)')
        plt.title('Топ-15 найважливіших ознак для прогнозування')
        plt.tight_layout()
        
        # Додаємо значення на барах
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{imp:.4f}', va='center', fontsize=9)
        
        plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n📊 Графік збережено: feature_importance_comparison.png")

def analyze_feature_changes(system, test_data):
    """Аналіз змін в ознаках для конкретного прикладу"""
    print("\n" + "="*80)
    print("🔄 АНАЛІЗ ЗМІН В ОЗНАКАХ")
    print("="*80)
    
    # Ключові ознаки, які могли вплинути на зниження
    critical_features = [
        'supplier_category_win_rate',
        'supplier_category_experience',
        'has_category_experience',
        'experience_type',
        'category_entry_barrier',
        'supplier_win_rate',
        'competitive_strength',
        'supplier_category_fit'
    ]
    
    supplier_profile = system.supplier_profiler.profiles.get(test_data['EDRPOU'])
    features = system.feature_extractor.extract_features(test_data, supplier_profile)
    
    print("\n⚠️ КРИТИЧНІ ОЗНАКИ ДЛЯ ВАШОГО ПРОГНОЗУ:")
    print("-"*80)
    
    for feat in critical_features:
        if feat in features:
            value = features[feat]
            
            # Інтерпретація
            interpretation = ""
            if feat == 'has_category_experience' and value == 0:
                interpretation = "❌ НЕМАЄ досвіду в категорії!"
            elif feat == 'experience_type':
                if value == 1:
                    interpretation = "✅ Прямий досвід в категорії"
                elif value == 2:
                    interpretation = "⚡ Досвід в кластері"
                else:
                    interpretation = "❌ Немає релевантного досвіду"
            elif feat == 'supplier_category_win_rate' and value < 0.2:
                interpretation = "⚠️ Низький win rate в категорії"
            elif feat == 'category_entry_barrier' and value > 0.7:
                interpretation = "⚠️ Високий бар'єр входу"
            
            print(f"{feat:35}: {value:8.4f}  {interpretation}")
    
    # Аналіз категорійного досвіду
    print("\n📋 ДОСВІД В КАТЕГОРІЇ:")
    category = test_data['F_INDUSTRYNAME']
    
    if supplier_profile:
        if category in supplier_profile.categories:
            cat_data = supplier_profile.categories[category]
            print(f"   ✅ Є досвід в '{category}':")
            print(f"      - Всього тендерів: {cat_data.get('total', 0)}")
            print(f"      - Виграно: {cat_data.get('won', 0)}")
            print(f"      - Win rate: {cat_data.get('win_rate', 0):.2%}")
        else:
            print(f"   ❌ НЕМАЄ досвіду в '{category}'")
            print(f"   📂 Категорії де є досвід:")
            for cat_name, cat_data in list(supplier_profile.categories.items())[:5]:
                print(f"      - {cat_name}: {cat_data.get('total', 0)} тендерів, "
                      f"win rate {cat_data.get('win_rate', 0):.2%}")

def suggest_improvements(features, predictions):
    """Пропозиції щодо покращення прогнозу"""
    print("\n" + "="*80)
    print("💡 РЕКОМЕНДАЦІЇ ДЛЯ ПОКРАЩЕННЯ ПРОГНОЗУ")
    print("="*80)
    
    ensemble_pred = predictions['ensemble_prediction']
    
    print(f"\nПоточний прогноз: {ensemble_pred:.2%}")
    
    if ensemble_pred < 0.4:
        print("\n⚠️ Низька ймовірність перемоги. Основні проблеми:")
        
        # Аналіз проблем
        problems = []
        
        if features.get('has_category_experience', 0) == 0:
            problems.append("1. Відсутність досвіду в категорії - це КРИТИЧНО!")
            problems.append("   Рекомендація: Розгляньте участь в категоріях де є досвід")
        
        if features.get('supplier_category_win_rate', 0) < 0.2:
            problems.append("2. Низький win rate в категорії")
            problems.append("   Рекомендація: Покращіть цінову пропозицію")
        
        if features.get('category_entry_barrier', 0) > 0.7:
            problems.append("3. Високий бар'єр входу в категорію")
            problems.append("   Рекомендація: Розгляньте менш конкурентні категорії")
        
        if features.get('supplier_vs_top_competitors', 0) < -0.2:
            problems.append("4. Значно слабші за топ конкурентів")
            problems.append("   Рекомендація: Проаналізуйте стратегії лідерів")
        
        for problem in problems:
            print(f"\n{problem}")
    
    # Потенціал покращення
    print("\n📈 ПОТЕНЦІАЛ ПОКРАЩЕННЯ:")
    
    if features.get('has_category_experience', 0) == 0:
        potential_gain = 0.15  # 15% від досвіду
        print(f"   • Набуття досвіду в категорії: +{potential_gain:.0%}")
    
    if features.get('supplier_win_rate', 0) < 0.3:
        potential_gain = 0.1
        print(f"   • Покращення загального win rate: +{potential_gain:.0%}")
    
    if features.get('has_brand', 0) == 0:
        potential_gain = 0.05
        print(f"   • Робота з брендовими товарами: +{potential_gain:.0%}")

def main():
    """Головна функція аналізу"""
    print("🚀 КОМПЛЕКСНИЙ АНАЛІЗ ПРОГНОЗУ ТЕНДЕРА")
    print("="*80)
    
    # Ініціалізація системи
    print("\n📦 Завантаження системи...")
    system = TenderAnalysisSystem(
        categories_file="categories.jsonl",
        qdrant_host="localhost",
        qdrant_port=6333
    )
    system.initialize_system()
    
    # Завантаження моделі
    model_file = "files/tender_system_trained.pkl"
    if Path(model_file).exists():
        system.load_system(model_file)
        print("✅ Модель завантажена")
    else:
        print("❌ Модель не знайдена!")
        return
    
    # Завантаження профілів
    profiles_file = "files/supplier_profiles_with_clusters.json"
    if Path(profiles_file).exists():
        system.supplier_profiler.load_profiles(profiles_file)
        print(f"✅ Завантажено {len(system.supplier_profiler.profiles)} профілів")
    
    # Тестові дані
    test_data = {
        "EDRPOU": "24366711", 
        "F_ITEMNAME": "Підшипник 6301-2RSH",
        "F_TENDERNAME": "Закупівля підшипників. Пропозиції надавати строго по заявкам !",
        "F_INDUSTRYNAME": "Подшипники"
    }
    
    # 1. Детальний аналіз прогнозу
    results = analyze_single_prediction(system, test_data)
    
    # 2. Завантаження та аналіз SHAP
    shap_report = load_shap_analysis()
    if shap_report:
        compare_feature_importance(shap_report)
    
    # 3. Аналіз змін в ознаках
    analyze_feature_changes(system, test_data)
    
    # 4. Рекомендації
    suggest_improvements(results['features'], results)
    
    # 5. Фінальний висновок
    print("\n" + "="*80)
    print("📝 ВИСНОВОК")
    print("="*80)
    
    print(f"\n🎯 Фінальний прогноз: {results['ensemble_prediction']:.2%}")
    
    if results['ensemble_prediction'] < 0.4:
        print("\n❌ ОСНОВНА ПРИЧИНА НИЗЬКОГО ПРОГНОЗУ:")
        print("   Відсутність досвіду в категорії 'Сільськогосподарські запчастини'")
        print("   Це найважливіший фактор згідно з SHAP аналізом!")
        print("\n💡 ЩО РОБИТИ:")
        print("   1. Використовуйте постачальника з досвідом в цій категорії")
        print("   2. Або виберіть категорію де постачальник має досвід")
        print("   3. Перевірте чи правильно вказана категорія в тестових даних")

if __name__ == "__main__":
    main()