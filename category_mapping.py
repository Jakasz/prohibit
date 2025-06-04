import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
import pandas as pd
from pathlib import Path

def analyze_categories_and_create_mapping(categories_file: str, output_file: str = "category_mappings.json"):
    """
    Аналіз категорій з JSONL файлу та створення маппінгу
    """
    print(f"📂 Завантаження категорій з {categories_file}...")
    
    # Завантаження всіх категорій
    categories = []
    with open(categories_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('active', True):  # Тільки активні категорії
                    categories.append(data.get('category', '').strip())
            except:
                continue
    
    print(f"✅ Завантажено {len(categories)} активних категорій")
    
    # Нормалізація та групування схожих категорій
    def normalize_text(text):
        """Базова нормалізація тексту"""
        text = text.lower().strip()
        # Видалення зайвих символів
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    # Створення груп схожих категорій
    category_groups = defaultdict(list)
    processed = set()
    
    print("🔄 Групування схожих категорій...")
    
    for i, cat1 in enumerate(categories):
        if cat1 in processed:
            continue
            
        norm_cat1 = normalize_text(cat1)
        group = [cat1]
        processed.add(cat1)
        
        # Пошук схожих категорій
        for cat2 in categories[i+1:]:
            if cat2 in processed:
                continue
                
            norm_cat2 = normalize_text(cat2)
            
            # Розрахунок схожості
            similarity = SequenceMatcher(None, norm_cat1, norm_cat2).ratio()
            
            # Додаткові перевірки схожості
            # 1. Одна категорія є підстрокою іншої
            if (norm_cat1 in norm_cat2 or norm_cat2 in norm_cat1) and len(norm_cat1) > 5:
                group.append(cat2)
                processed.add(cat2)
            # 2. Висока схожість (>80%)
            elif similarity > 0.8:
                group.append(cat2)
                processed.add(cat2)
            # 3. Спільні ключові слова
            elif len(set(norm_cat1.split()) & set(norm_cat2.split())) >= 2:
                group.append(cat2)
                processed.add(cat2)
        
        # Вибір канонічного імені (найкоротше або найчастіше)
        canonical = min(group, key=len)  # Беремо найкоротше
        category_groups[canonical] = group
    
    print(f"✅ Створено {len(category_groups)} груп категорій")
    
    # Створення фінального маппінгу
    mapping = {}
    
    # Додаємо базові категорії з ключовими словами
    base_mappings = {
        "Транспортні послуги": ["транспорт", "перевезення", "логістика", "доставка", "вантаж"],
        "ІТ послуги": ["програм", "software", "it", "інформацій", "комп'ютер", "цифров"],
        "Будівельні матеріали": ["будів", "констру", "цемент", "бетон", "цегла"],
        "Медичні товари": ["медич", "лікар", "фарма", "препарат", "здоров"],
        "Продукти харчування": ["продукт", "харч", "їжа", "food", "молоч", "м'яс"],
        "Офісні товари": ["офіс", "канцел", "папір", "ручк", "office"],
        "Паливо": ["палив", "бензин", "дизель", "газ", "нафт"],
        "Запчастини": ["запчаст", "деталь", "втулк", "підшипник", "ремонт"],
        "Послуги": ["послуг", "сервіс", "обслуг", "консульт", "service"]
    }
    
    # Розподіл груп по базовим категоріям
    for canonical, group in category_groups.items():
        matched = False
        norm_canonical = normalize_text(canonical)
        
        # Пошук відповідності базовим категоріям
        for base_cat, keywords in base_mappings.items():
            for keyword in keywords:
                if keyword in norm_canonical:
                    if base_cat not in mapping:
                        mapping[base_cat] = []
                    mapping[base_cat].extend(group)
                    matched = True
                    break
            if matched:
                break
        
        # Якщо не знайдено відповідність - створюємо нову категорію
        if not matched:
            mapping[canonical] = group
    
    # Видалення дублікатів
    for key in mapping:
        mapping[key] = list(set(mapping[key]))
    
    # Статистика
    print("\n📊 Статистика маппінгу:")
    sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[1]), reverse=True)
    for canonical, variants in sorted_mapping[:10]:
        print(f"  {canonical}: {len(variants)} варіантів")
    
    # Збереження
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Маппінг збережено в {output_file}")
    print(f"📈 Всього канонічних категорій: {len(mapping)}")
    print(f"📈 Всього варіантів: {sum(len(v) for v in mapping.values())}")
    
    return mapping

# Виклик функції
if __name__ == "__main__":
    mapping = analyze_categories_and_create_mapping("categories.jsonl")