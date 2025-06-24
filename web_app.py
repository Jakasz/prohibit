# app.py
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, session
import json
import logging
from datetime import datetime
from pathlib import Path
import os
from functools import wraps
import time

import numpy as np

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

@app.route('/profiles')
def profiles_page():
    """Сторінка з профілями постачальників"""
    return render_template('profiles.html')

@app.route('/api/profiles')
@require_initialized
def get_profiles():
    """API для отримання списку профілів"""
    try:
        # Параметри пагінації та фільтрації
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 50))
        search = request.args.get('search', '').strip()
        sort_by = request.args.get('sort_by', 'win_rate')
        sort_order = request.args.get('sort_order', 'desc')
        
        if not system.supplier_profiler or not hasattr(system.supplier_profiler, 'profiles'):
            return jsonify({
                'profiles': [],
                'total': 0,
                'page': page,
                'per_page': per_page
            })
        
        # Фільтрація профілів
        filtered_profiles = []
        for edrpou, profile in system.supplier_profiler.profiles.items():
            # Пошук по ЄДРПОУ або назві
            if search:
                if search not in edrpou and search.lower() not in profile.name.lower():
                    continue
            
            # Підготовка даних для відображення
            profile_data = {
                'edrpou': edrpou,
                'name': profile.name,
                'total_tenders': profile.metrics.total_tenders,
                'won_tenders': profile.metrics.won_tenders,
                'total_positions': profile.metrics.total_positions,
                'won_positions': profile.metrics.won_positions,
                'win_rate': round(profile.metrics.win_rate, 3),
                'position_win_rate': round(profile.metrics.position_win_rate, 3),
                'market_position': profile.market_position,
                'reliability_score': round(profile.reliability_score, 3),
                'specialization_score': round(profile.metrics.specialization_score, 3),
                'categories_count': len(profile.categories),
                'main_category': max(profile.categories.items(), key=lambda x: x[1].get('total', 0))[0] if profile.categories else 'Невідомо'
            }
            filtered_profiles.append(profile_data)
        
        # Сортування
        reverse = sort_order == 'desc'
        if sort_by in ['edrpou', 'name', 'main_category', 'market_position']:
            filtered_profiles.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)
        else:
            filtered_profiles.sort(key=lambda x: x.get(sort_by, 0), reverse=reverse)
        
        # Пагінація
        total = len(filtered_profiles)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_profiles = filtered_profiles[start:end]
        
        return jsonify({
            'profiles': paginated_profiles,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
        
    except Exception as e:
        logger.error(f"Помилка отримання профілів: {e}")
        return jsonify({
            'error': str(e),
            'profiles': [],
            'total': 0
        }), 500

@app.route('/api/profile/<edrpou>/detailed')
@require_initialized
def get_profile_detailed(edrpou):
    """Детальна інформація про профіль постачальника"""
    try:
        if not system.supplier_profiler or not hasattr(system.supplier_profiler, 'profiles'):
            return jsonify({'error': 'Профілі не завантажені'}), 404
        
        profile = system.supplier_profiler.profiles.get(edrpou)
        if not profile:
            return jsonify({'error': 'Постачальник не знайдений'}), 404
        
        # Підготовка детальної відповіді
        response = {
            'edrpou': edrpou,
            'name': profile.name,
            'metrics': {
                'total_tenders': profile.metrics.total_tenders,
                'won_tenders': profile.metrics.won_tenders,
                'total_positions': profile.metrics.total_positions,
                'won_positions': profile.metrics.won_positions,
                'win_rate': round(profile.metrics.win_rate, 3),
                'position_win_rate': round(profile.metrics.position_win_rate, 3),
                'recent_win_rate': round(profile.metrics.recent_win_rate, 3),
                'growth_rate': round(profile.metrics.growth_rate, 3),
                'stability_score': round(profile.metrics.stability_score, 3),
                'specialization_score': round(profile.metrics.specialization_score, 3),
                'competition_resistance': round(profile.metrics.competition_resistance, 3)
            },
            'market_position': profile.market_position,
            'reliability_score': round(profile.reliability_score, 3),
            'profile_version': profile.profile_version,
            'last_updated': profile.last_updated,
            'competitive_advantages': profile.competitive_advantages,
            'weaknesses': profile.weaknesses,
            'categories': {},
            'industries': {},
            'brand_expertise': profile.brand_expertise[:10] if hasattr(profile, 'brand_expertise') else [],
            'clusters': getattr(profile, 'clusters', []),
            'risk_indicators': getattr(profile, 'risk_indicators', {}),
            'has_risks': getattr(profile, 'has_risks', False),
            'overall_risk_level': getattr(profile, 'overall_risk_level', 'low')
        }
        
        # Категорії (топ 10)
        for cat_name, cat_data in sorted(
            profile.categories.items(), 
            key=lambda x: x[1].get('total', 0), 
            reverse=True
        )[:10]:
            response['categories'][cat_name] = {
                'total': cat_data.get('total', 0),
                'won': cat_data.get('won', 0),
                'win_rate': round(cat_data.get('win_rate', 0), 3),
                'revenue': round(cat_data.get('revenue', 0), 2),
                'specialization': round(cat_data.get('specialization', 0), 3)
            }
        
        # Індустрії (топ 5)
        if hasattr(profile, 'industries'):
            for ind_name, ind_data in sorted(
                profile.industries.items(),
                key=lambda x: x[1].get('total', 0),
                reverse=True
            )[:5]:
                response['industries'][ind_name] = {
                    'total': ind_data.get('total', 0),
                    'won': ind_data.get('won', 0),
                    'win_rate': round(ind_data.get('win_rate', 0), 3),
                    'revenue': round(ind_data.get('revenue', 0), 2)
                }
        
        # Топ конкуренти
        if hasattr(profile, 'top_competitors'):
            response['top_competitors'] = profile.top_competitors[:5]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Помилка отримання детального профілю: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/profiles/statistics')
@require_initialized
def get_profiles_statistics():
    """Загальна статистика по профілях"""
    try:
        if not system.supplier_profiler or not hasattr(system.supplier_profiler, 'profiles'):
            return jsonify({
                'total_suppliers': 0,
                'market_positions': {},
                'avg_win_rate': 0,
                'total_tenders': 0
            })
        
        # Збір статистики
        total_suppliers = len(system.supplier_profiler.profiles)
        market_positions = defaultdict(int)
        win_rates = []
        total_tenders = 0
        total_won = 0
        
        for profile in system.supplier_profiler.profiles.values():
            market_positions[profile.market_position] += 1
            win_rates.append(profile.metrics.win_rate)
            total_tenders += profile.metrics.total_tenders
            total_won += profile.metrics.won_tenders
        
        avg_win_rate = np.mean(win_rates) if win_rates else 0
        
        return jsonify({
            'total_suppliers': total_suppliers,
            'market_positions': dict(market_positions),
            'avg_win_rate': round(avg_win_rate, 3),
            'total_tenders': total_tenders,
            'total_won': total_won,
            'overall_win_rate': round(total_won / total_tenders, 3) if total_tenders > 0 else 0
        })
        
    except Exception as e:
        logger.error(f"Помилка отримання статистики: {e}")
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