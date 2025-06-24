# app.py
from flask import Flask, render_template, request, jsonify, session
import json
import logging
from datetime import datetime
from pathlib import Path
import os
from functools import wraps
import time

# Імпорт вашої системи
from tender_analysis_system import TenderAnalysisSystem

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Змініть на випадковий ключ

# Глобальна змінна для системи
system = None
initialization_status = {
    'is_initializing': False,
    'is_initialized': False,
    'error': None,
    'progress': 0,
    'message': ''
}

def initialize_system():
    """Ініціалізація системи при першому запуску"""
    global system, initialization_status
    
    try:
        initialization_status['is_initializing'] = True
        initialization_status['message'] = 'Початок ініціалізації системи...'
        initialization_status['progress'] = 10
        
        # Створення системи
        system = TenderAnalysisSystem(
            categories_file="categories.jsonl",
            qdrant_host="localhost",
            qdrant_port=6333
        )
        
        initialization_status['message'] = 'Ініціалізація підсистем...'
        initialization_status['progress'] = 30
        
        # Ініціалізація
        if not system.initialize_system():
            raise Exception("Помилка ініціалізації системи")
        
        initialization_status['message'] = 'Завантаження збереженої моделі...'
        initialization_status['progress'] = 50
        
        # Спробуємо завантажити збережену систему
        saved_models = [
            "files/tender_system_trained.pkl"
        ]
        
        model_loaded = False
        for model_file in saved_models:
            if Path(model_file).exists():
                try:
                    system.load_system(model_file)
                    logger.info(f"✅ Завантажено модель з {model_file}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Не вдалося завантажити {model_file}: {e}")
        
        if not model_loaded:
            initialization_status['message'] = 'Модель не знайдена. Потрібне навчання...'
            initialization_status['error'] = 'Модель не натренована. Запустіть exec_training.py'
            return False
        
        initialization_status['message'] = 'Завантаження профілів постачальників...'
        initialization_status['progress'] = 70
        
        # Завантаження профілів
        profile_files = [
            "files/supplier_profiles_with_clusters.json"            
        ]
        
        profiles_loaded = False
        for profile_file in profile_files:
            if Path(profile_file).exists():
                try:
                    system.supplier_profiler.load_profiles(profile_file)
                    logger.info(f"✅ Завантажено {len(system.supplier_profiler.profiles)} профілів")
                    profiles_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Не вдалося завантажити профілі: {e}")
        
        if not profiles_loaded:
            logger.warning("⚠️ Профілі постачальників не знайдено")
        
        initialization_status['message'] = 'Завантаження ринкової статистики...'
        initialization_status['progress'] = 90
        
        # Завантаження маркет статистики
        if hasattr(system, 'market_stats') and Path("market_statistics.json").exists():
            system.market_stats.load_statistics()
            logger.info("✅ Завантажено ринкову статистику")
        
        initialization_status['is_initialized'] = True
        initialization_status['is_initializing'] = False
        initialization_status['progress'] = 100
        initialization_status['message'] = 'Система готова до роботи!'
        
        logger.info("✅ Система повністю ініціалізована")
        return True
        
    except Exception as e:
        logger.error(f"❌ Помилка ініціалізації: {e}")
        initialization_status['is_initializing'] = False
        initialization_status['error'] = str(e)
        return False

def require_initialized(f):
    """Декоратор для перевірки ініціалізації системи"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not initialization_status['is_initialized']:
            return jsonify({
                'error': 'Система ще не ініціалізована',
                'status': initialization_status
            }), 503
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Головна сторінка"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Отримання статусу системи"""
    return jsonify({
        'initialized': initialization_status['is_initialized'],
        'initializing': initialization_status['is_initializing'],
        'progress': initialization_status['progress'],
        'message': initialization_status['message'],
        'error': initialization_status['error']
    })

@app.route('/api/predict', methods=['POST'])
@require_initialized
def predict():
    """API для прогнозування"""
    try:
        data = request.json
        
        # Валідація вхідних даних
        required_fields = ['edrpou', 'item_name', 'tender_name', 'industry_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Поле {field} є обов\'язковим'}), 400
        
        # Підготовка даних для прогнозування
        test_tender = {
            "EDRPOU": data['edrpou'],
            "F_ITEMNAME": data['item_name'],
            "F_TENDERNAME": data['tender_name'],
            "F_INDUSTRYNAME": data['industry_name']
        }
        
        # Додаткові поля якщо є
        if data.get('cpv'):
            test_tender['CPV'] = data['cpv']
        if data.get('budget'):
            test_tender['ITEM_BUDGET'] = float(data['budget'])
        
        # Отримання профілю постачальника
        supplier_profile = None
        if system.supplier_profiler and hasattr(system.supplier_profiler, 'profiles'):
            supplier_profile = system.supplier_profiler.profiles.get(data['edrpou'])
        
        # Прогнозування
        start_time = time.time()
        predictions = system.predictor.predict_tender(
            tender_items=[test_tender],
            supplier_profiles=system.supplier_profiler.profiles
        )
        prediction_time = time.time() - start_time
        
        # Обробка результатів
        if predictions and len(predictions) > 0:
            result = predictions[0]
            
            # Аналіз конкуренції (якщо доступно)
            competition_analysis = None
            if hasattr(system, 'competition_analyzer'):
                try:
                    competition_analysis = system.competition_analyzer.analyze_tender_competition(test_tender)
                except:
                    pass
            
            response = {
                'success': True,
                'prediction': {
                    'probability': result['probability'],
                    'confidence': result['confidence'],
                    'risk_factors': result.get('risk_factors', [])
                },
                'supplier_info': {
                    'name': supplier_profile.name if supplier_profile else 'Невідомий постачальник',
                    'edrpou': data['edrpou'],
                    'total_tenders': supplier_profile.metrics.total_tenders if supplier_profile else 0,
                    'win_rate': supplier_profile.metrics.win_rate if supplier_profile else 0,
                    'market_position': supplier_profile.market_position if supplier_profile else 'unknown'
                },
                'competition': competition_analysis if competition_analysis else None,
                'processing_time': round(prediction_time, 3)
            }
            
            # Додаткова інформація якщо є профіль
            if supplier_profile:
                response['supplier_info']['reliability_score'] = supplier_profile.reliability_score
                response['supplier_info']['specialization_score'] = supplier_profile.metrics.specialization_score
                
                # Досвід в категорії
                category_experience = supplier_profile.categories.get(data['industry_name'], {})
                if category_experience:
                    response['supplier_info']['category_experience'] = {
                        'total_tenders': category_experience.get('total', 0),
                        'win_rate': category_experience.get('win_rate', 0)
                    }
            
            return jsonify(response)
        else:
            return jsonify({
                'success': False,
                'error': 'Не вдалося отримати прогноз'
            }), 500
            
    except Exception as e:
        logger.error(f"Помилка прогнозування: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/categories')
@require_initialized
def get_categories():
    """Отримання списку категорій"""
    try:
        categories = []
        
        # Спробуємо отримати категорії з різних джерел
        if hasattr(system, 'categories_manager') and system.categories_manager:
            if hasattr(system.categories_manager, 'categories'):
                for cat_id, cat_data in system.categories_manager.categories.items():
                    if cat_data.get('active', True):
                        categories.append(cat_data.get('name', cat_id))
        
        # Якщо категорій немає, використаємо базовий список
        if not categories:
            categories = [
                "Сільськогосподарські запчастини",
                "Електроніка та електротехніка", 
                "Будівельні матеріали",
                "Канцелярські товари та офісне обладнання",
                "Медичні товари та обладнання",
                "Продукти харчування",
                "Паливо та мастильні матеріали",
                "Послуги",
                "Зв'язок та інтернет"
            ]
        
        return jsonify(sorted(categories))
        
    except Exception as e:
        logger.error(f"Помилка отримання категорій: {e}")
        return jsonify([])

@app.route('/api/supplier/<edrpou>')
@require_initialized
def get_supplier_info(edrpou):
    """Отримання інформації про постачальника"""
    try:
        if not system.supplier_profiler or not hasattr(system.supplier_profiler, 'profiles'):
            return jsonify({'error': 'Профілі не завантажені'}), 404
        
        profile = system.supplier_profiler.profiles.get(edrpou)
        if not profile:
            return jsonify({'error': 'Постачальник не знайдений'}), 404
        
        # Підготовка відповіді
        response = {
            'edrpou': edrpou,
            'name': profile.name,
            'metrics': {
                'total_tenders': profile.metrics.total_tenders,
                'won_tenders': profile.metrics.won_tenders,
                'win_rate': round(profile.metrics.win_rate, 3),
                'position_win_rate': round(profile.metrics.position_win_rate, 3),
                'stability_score': round(profile.metrics.stability_score, 3),
                'specialization_score': round(profile.metrics.specialization_score, 3)
            },
            'market_position': profile.market_position,
            'reliability_score': round(profile.reliability_score, 3),
            'categories': {},
            'competitive_advantages': profile.competitive_advantages,
            'weaknesses': profile.weaknesses
        }
        
        # Топ категорії
        for cat_name, cat_data in sorted(
            profile.categories.items(), 
            key=lambda x: x[1].get('total', 0), 
            reverse=True
        )[:5]:
            response['categories'][cat_name] = {
                'total': cat_data.get('total', 0),
                'won': cat_data.get('won', 0),
                'win_rate': round(cat_data.get('win_rate', 0), 3)
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Помилка отримання профілю: {e}")
        return jsonify({'error': str(e)}), 500

# --- Initialization before first request workaround ---
@app.before_request
def ensure_initialized():
    """Ensure system is initialized before handling any request (runs only once)."""
    if not initialization_status['is_initialized'] and not initialization_status['is_initializing']:
        import threading
        init_thread = threading.Thread(target=initialize_system)
        init_thread.daemon = True
        init_thread.start()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)