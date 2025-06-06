"""
Спрощений інтерфейс для прогнозування тендерів
"""
import json
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

class SimpleTenderPredictor:
    """Простий інтерфейс для швидкого прогнозування"""
    
    def __init__(self, system_path: str = "tender_system_improved.pkl"):
        """
        Args:
            system_path: шлях до збереженої системи
        """
        print("🚀 Завантаження системи...")
        
        # Завантаження системи
        from tender_analysis_system import TenderAnalysisSystem
        
        self.system = TenderAnalysisSystem()
        
        if Path(system_path).exists():
            self.system.load_system(system_path)
            print("✅ Система завантажена")
        else:
            print("⚠️ Збережену систему не знайдено. Ініціалізація нової...")
            self.system.initialize_system()
            # Тут потрібно буде натренувати модель
            raise RuntimeError("Потрібно спочатку натренувати модель!")
        
        # Перевірка готовності
        if not self.system.is_trained:
            raise RuntimeError("Модель не натренована!")
            
    def prepare_input_data(self, simplified_data: List[Dict]) -> List[Dict]:
        """
        Конвертація спрощеного формату в повний
        
        Args:
            simplified_data: список словників з полями:
                - edrpou: код ЄДРПОУ
                - tender_name: назва тендера
                - item_name: назва товару/послуги
                - industry_name: назва галузі
                - cpv: код CPV (опціонально)
                - budget: бюджет (опціонально)
        """
        full_data = []
        
        for idx, item in enumerate(simplified_data):
            # Створюємо повний формат
            full_item = {
                'F_TENDERNUMBER': f"TEMP_{idx}_{datetime.now().strftime('%Y%m%d')}",
                'EDRPOU': str(item.get('edrpou', '')),
                'F_TENDERNAME': item.get('tender_name', ''),
                'F_ITEMNAME': item.get('item_name', ''),
                'F_INDUSTRYNAME': item.get('industry_name', ''),
                'CPV': item.get('cpv', 0),
                'ITEM_BUDGET': float(item.get('budget', 0)) if item.get('budget') else 0,
                'DATEEND': datetime.now().strftime('%d.%m.%Y'),
                'WON': False  # За замовчуванням
            }
            
            # Додаткові поля якщо є
            if 'supplier_name' in item:
                full_item['supp_name'] = item['supplier_name']
            
            full_data.append(full_item)
            
        return full_data
    
    def predict_batch(self, 
                     input_data: List[Dict], 
                     threshold: float = 0.6,
                     include_all: bool = False) -> Dict[str, Any]:
        """
        Прогнозування для батчу тендерів
        
        Args:
            input_data: список тендерів у спрощеному форматі
            threshold: поріг ймовірності (0.6 = 60%)
            include_all: якщо True, повертає всі результати незалежно від порогу
            
        Returns:
            Словник з результатами прогнозування
        """
        print(f"\n🔮 Прогнозування для {len(input_data)} тендерів...")
        
        # 1. Підготовка даних
        full_data = self.prepare_input_data(input_data)
        
        # 2. Отримання профілів постачальників
        supplier_profiles = {}
        for item in full_data:
            edrpou = item.get('EDRPOU')
            if edrpou and edrpou not in supplier_profiles:
                profile = self.system.supplier_profiler.get_profile(edrpou)
                if profile:
                    supplier_profiles[edrpou] = profile.to_dict()
        
        # 3. Прогнозування
        predictions = self.system.predictor.predict_tender(full_data, supplier_profiles)
        
        # 4. Обробка результатів
        results = {
            'total_items': len(input_data),
            'threshold': threshold,
            'high_probability': [],
            'all_predictions': [],
            'summary': {
                'above_threshold': 0,
                'below_threshold': 0,
                'average_probability': 0
            }
        }
        
        total_prob = 0
        
        for idx, (orig_item, pred) in enumerate(zip(input_data, predictions)):
            # Додаємо оригінальні дані до результату
            result_item = {
                'index': idx,
                'edrpou': orig_item.get('edrpou'),
                'tender_name': orig_item.get('tender_name'),
                'item_name': orig_item.get('item_name'),
                'industry': orig_item.get('industry_name'),
                'cpv': orig_item.get('cpv'),
                'probability': pred['probability'],
                'confidence': pred['confidence'],
                'risk_factors': pred.get('risk_factors', [])
            }
            
            total_prob += pred['probability']
            
            # Фільтрація за порогом
            if pred['probability'] >= threshold:
                results['high_probability'].append(result_item)
                results['summary']['above_threshold'] += 1
            else:
                results['summary']['below_threshold'] += 1
            
            # Всі результати (якщо потрібно)
            if include_all:
                results['all_predictions'].append(result_item)
        
        # Підсумкова статистика
        results['summary']['average_probability'] = total_prob / len(predictions) if predictions else 0
        
        return results
    
    def predict_from_file(self, 
                         file_path: str, 
                         threshold: float = 0.6,
                         output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Прогнозування з файлу JSON або JSONL
        
        Args:
            file_path: шлях до файлу з даними
            threshold: поріг ймовірності
            output_path: куди зберегти результати (опціонально)
        """
        print(f"📂 Завантаження даних з {file_path}")
        
        # Визначення формату файлу
        data = []
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError("Підтримуються тільки .json та .jsonl файли")
        
        print(f"✅ Завантажено {len(data)} записів")
        
        # Прогнозування
        results = self.predict_batch(data, threshold=threshold, include_all=True)
        
        # Збереження результатів
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Результати збережено в {output_path}")
        
        return results
    
    def print_results(self, results: Dict[str, Any], detailed: bool = False):
        """Красивий вивід результатів"""
        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТИ ПРОГНОЗУВАННЯ")
        print("="*60)
        
        summary = results['summary']
        print(f"\n📈 Загальна статистика:")
        print(f"   • Всього тендерів: {results['total_items']}")
        print(f"   • Поріг: {results['threshold']*100:.0f}%")
        print(f"   • Вище порогу: {summary['above_threshold']} ({summary['above_threshold']/results['total_items']*100:.1f}%)")
        print(f"   • Нижче порогу: {summary['below_threshold']}")
        print(f"   • Середня ймовірність: {summary['average_probability']*100:.1f}%")
        
        if results['high_probability']:
            print(f"\n✅ Тендери з високою ймовірністю перемоги (>={results['threshold']*100:.0f}%):")
            print("-"*60)
            
            for item in sorted(results['high_probability'], key=lambda x: x['probability'], reverse=True):
                print(f"\n#{item['index']+1} | Ймовірність: {item['probability']*100:.1f}% ({item['confidence']})")
                print(f"   ЄДРПОУ: {item['edrpou']}")
                print(f"   Тендер: {item['tender_name'][:50]}...")
                print(f"   Товар: {item['item_name'][:50]}...")
                print(f"   Галузь: {item['industry']}")
                
                if item['risk_factors']:
                    print(f"   ⚠️ Ризики: {', '.join(item['risk_factors'])}")
        
        if detailed and results.get('all_predictions'):
            print(f"\n📋 Всі результати:")
            print("-"*60)
            for item in results['all_predictions']:
                status = "✅" if item['probability'] >= results['threshold'] else "❌"
                print(f"{status} #{item['index']+1}: {item['probability']*100:.1f}% - {item['item_name'][:40]}...")
        
        print("\n" + "="*60)

# Приклад використання
def example_usage():
    """Приклад використання"""
    
    # 1. Підготовка даних
    test_data = [
        {
            "edrpou": "12345678",
            "tender_name": "Закупівля запчастин для тракторів",
            "item_name": "Фільтр паливний для John Deere",
            "industry_name": "Сільське господарство",
            "cpv": "16810000",
            "budget": 50000
        },
        {
            "edrpou": "12345678",
            "tender_name": "Закупівля офісної техніки",
            "item_name": "Картриджі для принтерів HP",
            "industry_name": "Офісне обладнання",
            "cpv": "30125110"
        },
        # Додайте більше тендерів...
    ]
    
    # 2. Створення предиктора
    predictor = SimpleTenderPredictor("tender_system.pkl")
    
    # 3. Прогнозування
    results = predictor.predict_batch(
        test_data, 
        threshold=0.6,  # 60%
        include_all=True
    )
    
    # 4. Вивід результатів
    predictor.print_results(results, detailed=True)
    
    # 5. Збереження результатів
    with open("predictions_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    example_usage()