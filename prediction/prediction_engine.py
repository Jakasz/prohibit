import json
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from typing import Dict, List, Tuple, Optional, Any
import pickle
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

from profiles.supplier_profiler import SupplierProfile

# Опціональні імпорти
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Локальні імпорти (додати після створення файлів)
from features.feature_extractor import FeatureExtractor
from features.feature_processor import AdvancedFeatureProcessor
from AI.model_monitor import ModelMonitor

class PredictionEngine:
    """Основний двигун прогнозування тендерів"""

    def __init__(self, supplier_profiler, categories_manager):
        self.supplier_profiler = supplier_profiler        
        self.categories_manager = categories_manager
        self.feature_extractor = FeatureExtractor(categories_manager)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.monitor = ModelMonitor()        
        self.logger = logging.getLogger(__name__)

        # Конфігурація моделей
        self.model_configs = {
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 20,
                    'min_samples_leaf': 15,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 250,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'subsample': 0.7,
                    'random_state': 42
                }
            }
        }
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'class': XGBClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.7,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 10,
                    'gamma': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'eval_metric': 'logloss',
                    'tree_method': 'hist'
                }
            }


        self.ensemble_weights = {}
        self.calibrated_models = {}

    def analyze_with_shap(self, X_sample=None, sample_size=1000, save_plots=True):
        """
        Повний SHAP аналіз моделі
        
        Args:
            X_sample: дані для аналізу (якщо None - візьме з training_data)
            sample_size: кількість прикладів для аналізу
            save_plots: чи зберігати графіки
        """
        try:
            import shap
            
            
            self.logger.info("🔍 Початок SHAP аналізу...")
            
            # Підготовка даних
            if X_sample is None:
                if not hasattr(self, 'training_data') or self.training_data is None:
                    raise ValueError("Немає даних для аналізу. Спочатку натренуйте модель.")
                X, y = self.training_data
                # Беремо випадкову вибірку
                if len(X) > sample_size:
                    indices = np.random.choice(len(X), sample_size, replace=False)
                    X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
                    y_sample = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
                else:
                    X_sample = X
                    y_sample = y
            
            # Обробка даних через feature processor якщо є
            if hasattr(self, 'feature_processor'):
                X_sample_processed = self.feature_processor.transform(X_sample)
                X_sample_processed = self.feature_extractor.create_interaction_features(X_sample_processed)
            else:
                X_sample_processed = X_sample
            
            # Масштабування
            X_sample_scaled = self.scalers['main'].transform(X_sample_processed)
            
            # Аналіз для кожної моделі
            shap_results = {}
            
            for model_name, model in self.models.items():
                self.logger.info(f"\n📊 SHAP аналіз для {model_name}...")
                
                try:
                    # Створення пояснювача
                    if model_name == 'random_forest':
                        explainer = shap.TreeExplainer(model)
                    elif model_name == 'gradient_boosting':
                        explainer = shap.TreeExplainer(model)
                    elif model_name == 'xgboost' and hasattr(model, 'get_booster'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        # Для інших моделей використовуємо KernelExplainer
                        explainer = shap.KernelExplainer(
                            model.predict_proba, 
                            shap.sample(X_sample_scaled, 100)
                        )
                    
                    # Обчислення SHAP values
                    shap_values = explainer.shap_values(X_sample_scaled)
                    
                    # Для бінарної класифікації беремо значення для класу 1
                    if isinstance(shap_values, list) and len(shap_values) == 2:
                        shap_values = shap_values[1]
                    
                    # Зберігаємо результати
                    shap_results[model_name] = {
                        'shap_values': shap_values,
                        'explainer': explainer,
                        'expected_value': explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                    }
                    
                    # Створення візуалізацій
                    if save_plots:
                        self._create_shap_plots(
                            shap_values, 
                            X_sample_processed, 
                            model_name,
                            explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value
                        )
                    
                    # Аналіз experience_type якщо є
                    if 'experience_type' in self.feature_names:
                        self._analyze_experience_type_shap(shap_values, X_sample_processed, model_name)
                    
                except Exception as e:
                    self.logger.error(f"Помилка SHAP аналізу для {model_name}: {e}")
                    continue
            
            # Зберігаємо SHAP результати
            self.shap_results = shap_results
            
            # Створення загального звіту
            self._create_shap_report(shap_results, X_sample_processed)
            
            self.logger.info("✅ SHAP аналіз завершено!")
            
            return shap_results
            
        except ImportError:
            self.logger.error("❌ SHAP не встановлено. Виконайте: pip install shap")
            raise

    def _create_shap_plots(self, shap_values, X_sample, model_name, expected_value):
        """Створення SHAP візуалізацій"""
        import shap
        from pathlib import Path
        
        # Створюємо директорію для графіків
        plots_dir = Path("shap_plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Summary plot - загальний огляд
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig(plots_dir / f'shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(plots_dir / f'shap_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Dependence plots для топ-5 ознак
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        
        for idx in top_features_idx:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                idx, 
                shap_values, 
                X_sample,
                feature_names=self.feature_names,
                show=False
            )
            feature_name = self.feature_names[idx]
            plt.title(f'SHAP Dependence Plot - {feature_name} ({model_name})')
            plt.tight_layout()
            plt.savefig(plots_dir / f'shap_dependence_{model_name}_{feature_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"📈 Графіки збережено в {plots_dir}/")

    def _analyze_experience_type_shap(self, shap_values, X_sample, model_name):
        """Спеціальний аналіз для experience_type"""
        if 'experience_type' not in self.feature_names:
            return
        
        exp_type_idx = self.feature_names.index('experience_type')
        
        print(f"\n=== Аналіз experience_type для {model_name} ===")
        
        # Групуємо SHAP values по типах досвіду
        X_df = pd.DataFrame(X_sample, columns=self.feature_names)
        
        for exp_type in [1, 2, 3]:
            mask = X_df['experience_type'] == exp_type
            if mask.any():
                shap_mean = shap_values[mask, exp_type_idx].mean()
                shap_std = shap_values[mask, exp_type_idx].std()
                count = mask.sum()
                
                type_names = {1: "Прямий досвід", 2: "Кластерний досвід", 3: "Загальний"}
                print(f"\n{type_names[exp_type]}:")
                print(f"  Кількість: {count}")
                print(f"  Середній SHAP: {shap_mean:+.4f}")
                print(f"  Std SHAP: {shap_std:.4f}")
                print(f"  Вплив: {'Позитивний' if shap_mean > 0 else 'Негативний'}")

    def test_prediction_with_explanation(self, edrpou, item_name, category):
        """Швидкий тест прогнозу з поясненням"""
        
        test_data = {
            "EDRPOU": edrpou,
            "F_ITEMNAME": item_name,
            "F_TENDERNAME": f"Тестова закупівля {category}",
            "F_INDUSTRYNAME": category
        }
        
        print(f"\n🧪 ТЕСТ ПРОГНОЗУ")
        print("="*60)
        
        try:
            # Знаходимо профіль постачальника
            supplier_profile = None
            if hasattr(self, 'supplier_analyzer'):
                supplier_profile = self.supplier_analyzer.get_supplier_profile(edrpou)
                if supplier_profile:
                    print(f"✅ Знайдено профіль постачальника:")
                    print(f"   Загальний win rate: {supplier_profile.metrics.win_rate:.2%}")
                    print(f"   Досвід: {supplier_profile.metrics.total_tenders} тендерів")
                else:
                    print(f"⚠️ Профіль постачальника не знайдено")
            
            # Робимо прогноз
            result = self.explain_single_prediction(test_data, supplier_profile, show_plot=False)
            
            return result
            
        except Exception as e:
            print(f"❌ Помилка: {e}")
            import traceback
            traceback.print_exc()
            return None





    def _create_shap_report(self, shap_results, X_sample):
        """Створення детального SHAP звіту"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(X_sample),
            'models_analyzed': list(shap_results.keys()),
            'feature_analysis': {}
        }
        
        # Аналіз по кожній ознаці
        for feature_idx, feature_name in enumerate(self.feature_names):
            feature_report = {}
            
            for model_name, results in shap_results.items():
                shap_vals = results['shap_values'][:, feature_idx]
                
                feature_report[model_name] = {
                    'mean_abs_shap': float(np.abs(shap_vals).mean()),
                    'mean_shap': float(shap_vals.mean()),
                    'std_shap': float(shap_vals.std()),
                    'positive_impact_ratio': float((shap_vals > 0).mean()),
                    'feature_importance_rank': None  # Заповнимо пізніше
                }
            
            report['feature_analysis'][feature_name] = feature_report
        
        # Додаємо ранги важливості
        for model_name in shap_results.keys():
            # Сортуємо ознаки за важливістю
            importance_list = [
                (feat, data[model_name]['mean_abs_shap']) 
                for feat, data in report['feature_analysis'].items()
            ]
            importance_list.sort(key=lambda x: x[1], reverse=True)
            
            # Присвоюємо ранги
            for rank, (feat, _) in enumerate(importance_list, 1):
                report['feature_analysis'][feat][model_name]['feature_importance_rank'] = rank
        
        # Зберігаємо звіт
        with open('shap_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info("📄 SHAP звіт збережено в shap_analysis_report.json")
        
        # Виводимо топ-10 важливих ознак
        print("\n📊 ТОП-10 НАЙВАЖЛИВІШИХ ОЗНАК (за SHAP):")
        print("="*70)
        
        for model_name in shap_results.keys():
            print(f"\n{model_name}:")
            top_features = sorted(
                report['feature_analysis'].items(),
                key=lambda x: x[1][model_name]['mean_abs_shap'],
                reverse=True
            )[:10]
            
            for feat_name, feat_data in top_features:
                model_data = feat_data[model_name]
                direction = "↑" if model_data['mean_shap'] > 0 else "↓"
                print(f"  {feat_name:35} | SHAP: {model_data['mean_abs_shap']:.4f} | Напрям: {direction}")

    def explain_single_prediction(self, input_data, supplier_profile=None, show_plot=True):
        """
        Пояснення одного конкретного прогнозу
        
        Args:
            input_data: сирі дані (dict або DataFrame з полями EDRPOU, F_ITEMNAME тощо)
            supplier_profile: профіль постачальника (опціонально)
            show_plot: чи показувати waterfall plot
        """
        import shap
        
        if not hasattr(self, 'shap_results'):
            raise ValueError("Спочатку запустіть analyze_with_shap()")
        
        # 1. Конвертуємо в правильний формат
        if isinstance(input_data, pd.DataFrame):
            input_dict = input_data.iloc[0].to_dict()
        else:
            input_dict = input_data
        
        # 2. Отримуємо EDRPOU для пошуку профілю
        edrpou = input_dict.get('EDRPOU', '')
        
        # 3. Якщо профіль не переданий, намагаємось знайти
        if supplier_profile is None and edrpou and hasattr(self, 'supplier_analyzer'):
            supplier_profile = self.supplier_analyzer.get_supplier_profile(edrpou)
        
        # 4. Витягуємо ознаки через feature_extractor
        features = self.feature_extractor.extract_features(input_dict, supplier_profile)
        
        # 5. Створюємо DataFrame з правильними назвами колонок
        X_single = pd.DataFrame([features], columns=self.feature_names)
        
        # 6. Обробка даних
        if hasattr(self, 'feature_processor'):
            X_processed = self.feature_processor.transform(X_single)
            X_processed = self.feature_extractor.create_interaction_features(X_processed)
        else:
            X_processed = X_single
        
        X_scaled = self.scalers['main'].transform(X_processed)
        
        # 7. Прогноз
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict_proba(X_scaled)[0][1]
            predictions[model_name] = pred
        
        ensemble_pred = np.mean(list(predictions.values()))
        
        print(f"\n🎯 Прогноз: {ensemble_pred:.2%} ймовірність перемоги")
        print("="*70)
        
        # 8. Виводимо значення ключових ознак
        print("\n📋 Вхідні дані:")
        print(f"  EDRPOU: {edrpou}")
        print(f"  Назва позиції: {input_dict.get('F_ITEMNAME', 'N/A')}")
        print(f"  Категорія: {input_dict.get('F_INDUSTRYNAME', 'N/A')}")
        
        print("\n📊 Витягнуті ознаки (топ-10):")
        important_features = ['supplier_category_win_rate', 'supplier_win_rate', 
                            'experience_type', 'has_category_experience',
                            'supplier_experience', 'category_win_probability',
                            'has_brand', 'supplier_stability', 
                            'competitive_strength', 'supplier_vs_market_avg']
        
        for feat in important_features:
            if feat in features:
                print(f"  {feat:35}: {features[feat]:.4f}")
        
        # 9. Пояснення для кожної моделі
        for model_name, results in self.shap_results.items():
            print(f"\n📊 Пояснення від {model_name}:")
            print(f"Прогноз моделі: {predictions[model_name]:.2%}")
            
            # Отримуємо SHAP values для цього прикладу
            explainer = results['explainer']
            shap_values = explainer.shap_values(X_scaled)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Для класу 1
            
            # Сортуємо ознаки за впливом
            feature_impact = list(zip(self.feature_names, shap_values[0], X_processed.iloc[0]))
            feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("\nТоп-10 факторів:")
            for feat_name, impact, feat_value in feature_impact[:10]:
                direction = "збільшує" if impact > 0 else "зменшує"
                print(f"  {feat_name:30} = {feat_value:8.3f} | SHAP: {impact:+.4f} ({direction})")
            
            # Waterfall plot для візуалізації
            if show_plot and model_name == 'random_forest':  # Тільки для однієї моделі
                try:
                    import matplotlib.pyplot as plt
                    shap.plots.waterfall(
                        shap.Explanation(
                            values=shap_values[0],
                            base_values=results['expected_value'],
                            data=X_processed.iloc[0],
                            feature_names=self.feature_names
                        ),
                        max_display=15
                    )
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Не вдалось створити waterfall plot: {e}")
        
        return {
            'prediction': ensemble_pred,
            'model_predictions': predictions,
            'features': features,
            'top_factors': feature_impact[:10]
        }





    def train_model(self, validation_split: float = 0.2, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Обгортка для train_models з правильною назвою методу
        """
        # Спочатку потрібно підготувати дані
        if not hasattr(self, 'training_data') or self.training_data is None:
            raise ValueError("Немає даних для тренування. Спочатку викличте prepare_training_data()")
        
        # Розпакування даних
        X, y = self.training_data
        
        # Тренування моделей
        performance = self.train_models(X, y, test_size=validation_split)
        
        # Формування результату
        return {
            'performance_metrics': {
                'auc_score': performance.get('ensemble', {}).get('test_auc', 0),
                'model_scores': {
                    name: metrics.get('test_auc', 0) 
                    for name, metrics in performance.items() 
                    if name != 'ensemble'
                }
            },
            'feature_importance': self.feature_importance,
            'training_samples': len(X),
            'models_trained': list(self.models.keys())
        }

    def prepare_training_data_from_history(self, historical_data: List[Dict], supplier_profiles: Dict[str, SupplierProfile]):
        """
        Підготовка даних для тренування з історичних даних і профілів
        """
        # Конвертація профілів в словники
        profiles_dict = {}
        for edrpou, profile in supplier_profiles.items():
            if isinstance(profile, SupplierProfile):
                profiles_dict[edrpou] = profile.to_dict()
            else:
                profiles_dict[edrpou] = profile
        
        # Підготовка даних
        X, y = self.prepare_training_data(historical_data, profiles_dict)
        
        # Збереження для подальшого використання
        self.training_data = (X, y)
        
        return X, y

    def export_state(self) -> Dict[str, Any]:
        """Експорт стану предиктора"""
        return {
            'models': self.models,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'ensemble_weights': self.ensemble_weights,
            'feature_names': self.feature_names
        }

    def load_state(self, state_data: Dict[str, Any]):
        """Завантаження стану предиктора"""
        self.models = state_data.get('models', {})
        self.scalers = state_data.get('scalers', {})
        self.feature_importance = state_data.get('feature_importance', {})
        self.model_performance = state_data.get('model_performance', {})
        self.ensemble_weights = state_data.get('ensemble_weights', {})
        self.feature_names = state_data.get('feature_names', [])
        
        # Встановлення прапорця натренованості
        self.is_trained = len(self.models) > 0


    def update_actual_outcomes(self, outcomes: List[Dict[str, Any]]):
        """Оновлення фактичних результатів для моніторингу"""
        for outcome in outcomes:
            self.monitor.update_actual_outcome(
                tender_id=outcome['tender_number'],
                actual=outcome['won']
            )
        
        # Перевірка необхідності перенавчання
        should_retrain, info = self.monitor.should_retrain()
        if should_retrain:
            self.logger.warning(f"Рекомендується перенавчання моделі: {info['reasons']}")
        
        return info

    def prepare_training_data(self, historical_data: List[Dict], supplier_profiles: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        self.logger.info("Підготовка тренувальних даних...")
        features_list = []
        targets = []
        for item in historical_data:
            edrpou = item.get('EDRPOU')
            supplier_profile = supplier_profiles.get(edrpou)
            features = self.feature_extractor.extract_features(item, supplier_profile)
            features_list.append(features)
            targets.append(int(item.get('WON', 0)))
        X = pd.DataFrame(features_list)
        y = pd.Series(targets)
        self.feature_names = list(X.columns)
        X = X.fillna(0)
        self.logger.info(f"Підготовлено {len(X)} зразків з {len(X.columns)} фічами")
        self.logger.info(f"Розподіл класів: {y.value_counts().to_dict()}")
        return X, y
    

    def _optimize_ensemble_weights_advanced(self, X_test, y_test):
        """Покращена оптимізація ваг ансамблю"""
        try:
            from scipy.optimize import minimize
            
            # Збираємо прогнози від всіх моделей
            predictions = {}
            X_test_scaled = self.scalers['main'].transform(X_test)
            
            for name, model in self.models.items():
                predictions[name] = model.predict_proba(X_test_scaled)[:, 1]
            
            # Функція оптимізації
            def objective(weights):
                weighted_pred = np.zeros(len(y_test))
                for i, name in enumerate(self.models.keys()):
                    weighted_pred += weights[i] * predictions[name]
                return -roc_auc_score(y_test, weighted_pred)
            
            # Обмеження та початкові значення
            constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
            bounds = [(0, 1) for _ in range(len(self.models))]
            initial_weights = [1/len(self.models)] * len(self.models)
            
            # Оптимізація
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                self.ensemble_weights = dict(zip(self.models.keys(), result.x))
                self.logger.info(f"Оптимізовані ваги: {self.ensemble_weights}")
            else:
                # Fallback до рівних ваг
                self._optimize_ensemble_weights(X_test, y_test)
                
        except ImportError:
            self.logger.warning("scipy не встановлено, використовуємо рівні ваги")
            self._optimize_ensemble_weights(X_test, y_test)    

    def _calculate_sample_weights(self, y: pd.Series) -> np.ndarray:
        """Розрахунок ваг для збалансування класів"""
        from sklearn.utils.class_weight import compute_sample_weight
        
        # Автоматичний розрахунок ваг
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y
        )
        
        return sample_weights

    def _optimize_ensemble_weights_advanced(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Покращена оптимізація ваг ансамблю"""
        try:
            from scipy.optimize import minimize
            
            # Масштабування для прогнозування
            X_test_scaled = self.scalers['main'].transform(X_test)
            
            # Збираємо прогнози від всіх моделей
            predictions = {}
            for name, model in self.models.items():
                predictions[name] = model.predict_proba(X_test_scaled)[:, 1]
            
            # Функція для оптимізації
            def objective(weights):
                weighted_pred = np.zeros(len(y_test))
                for i, name in enumerate(self.models.keys()):
                    weighted_pred += weights[i] * predictions[name]
                return -roc_auc_score(y_test, weighted_pred)
            
            # Обмеження: сума ваг = 1, всі ваги >= 0
            constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
            bounds = [(0, 1) for _ in range(len(self.models))]
            initial_weights = [1/len(self.models)] * len(self.models)
            
            # Оптимізація
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                self.ensemble_weights = dict(zip(self.models.keys(), result.x))
                self.logger.info(f"Оптимізовані ваги ансамблю: {self.ensemble_weights}")
            else:
                self.logger.warning("Оптимізація не вдалася, використовуємо рівні ваги")
                self._optimize_ensemble_weights(X_test, y_test)
                
        except ImportError:
            self.logger.warning("scipy не встановлено, використовуємо рівні ваги")
            self._optimize_ensemble_weights(X_test, y_test)
        except Exception as e:
            self.logger.error(f"Помилка оптимізації: {e}")
            self._optimize_ensemble_weights(X_test, y_test)


    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, use_calibration: bool = True):
        self.logger.info("Початок тренування моделей...")
        
        # [1] Розділення на train/test (БЕЗ ЗМІН)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # ===== [2] НОВЕ: Feature Processing =====
        # Імпорт на початку файлу: from .feature_processor import AdvancedFeatureProcessor
        
        # Створення та навчання процесора
        self.feature_processor = AdvancedFeatureProcessor()
        X_train_processed = self.feature_processor.fit_transform(X_train, y_train)
        X_test_processed = self.feature_processor.transform(X_test)
        
        # Створення interaction features
        X_train_processed = self.feature_extractor.create_interaction_features(X_train_processed)
        X_test_processed = self.feature_extractor.create_interaction_features(X_test_processed)
        
        # Оновлення списку ознак
        self.feature_names = list(X_train_processed.columns)
        
        # ===== [3] Масштабування (ВИКОРИСТОВУЄМО ОБРОБЛЕНІ ДАНІ) =====
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)  
        X_test_scaled = scaler.transform(X_test_processed)        
        self.scalers['main'] = scaler

        # ===== [4] Цикл тренування моделей =====
        for model_name, config in self.model_configs.items():
            self.logger.info(f"Тренування моделі: {model_name}")
            
            # [5] Створення моделі
            model_class = config['class']
            model_params = config['params'].copy()  # Копія для модифікації
            
            # ===== [6]: Спеціальна обробка для різних моделей =====
            if model_name == 'xgboost' and XGBOOST_AVAILABLE:
                # Розрахунок ваги для балансування класів
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                model_params['scale_pos_weight'] = scale_pos_weight
                self.logger.info(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
                
            elif model_name == 'gradient_boosting':
                # Для GradientBoosting можна додати sample_weight при fit
                # Зберігаємо для використання пізніше
                self._gb_sample_weights = self._calculate_sample_weights(y_train)
            
            # Створення моделі з оновленими параметрами
            model = model_class(**model_params)
            
            # [7] Калібрація (БЕЗ ЗМІН)
            if use_calibration:
                model = CalibratedClassifierCV(model, cv=3)
            
            # ===== [8] Тренування (З УРАХУВАННЯМ ОСОБЛИВОСТЕЙ) =====
            if model_name == 'gradient_boosting' and hasattr(self, '_gb_sample_weights'):
                # Спеціальне тренування для GradientBoosting
                if use_calibration:
                    # CalibratedClassifierCV не підтримує sample_weight напряму
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled, y_train, sample_weight=self._gb_sample_weights)
            else:
                # Звичайне тренування
                model.fit(X_train_scaled, y_train)
            
            # Збереження моделі
            self.models[model_name] = model
            
            # Оцінка
            perf = self._evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name)
            self.model_performance[model_name] = perf
            
            # Аналіз важливості ознак
            self._analyze_feature_importance(model, self.feature_names, model_name)
            
            # ===== НОВЕ: Логування в монітор =====
            if hasattr(self, 'monitor'):
                self.monitor.performance_history.append({
                    'model': model_name,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': perf
                })

        # ===== [9] ЗАМІНА: Покращена оптимізація ваг ансамблю =====      
        # Використовуємо:
        self._optimize_ensemble_weights_advanced(X_test_processed, y_test)  
        
        # Оцінка ансамблю
        ensemble_pred = self.predict_proba_ensemble(X_test_processed) 
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        self.logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
        self.model_performance['ensemble'] = {'test_auc': ensemble_auc}

        # ===== Збереження додаткової інформації =====
        self.training_info = {
            'date': datetime.now().isoformat(),
            'n_samples': len(X_train),
            'n_features': len(self.feature_names),
            'class_distribution': {
                'train': dict(y_train.value_counts()),
                'test': dict(y_test.value_counts())
            }
        }
        
        # Оновлення версії моделі
        self.model_performance['version'] = self.model_performance.get('version', '1.0')
        
        return self.model_performance

    def _evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name: str) -> Dict:
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_test_pred = model.predict_proba(X_test)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        y_test_binary = (y_test_pred >= best_threshold).astype(int)
        report = classification_report(y_test, y_test_binary, output_dict=True)
        performance = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'best_threshold': best_threshold,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score']
        }
        return performance

    def _analyze_feature_importance(self, model, feature_names, model_name: str):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'base_estimator_') and hasattr(model.base_estimator_, 'feature_importances_'):
            importances = model.base_estimator_.feature_importances_
        else:
            importances = None
        if importances is not None:
            self.feature_importance[model_name] = dict(zip(feature_names, importances))

    def _optimize_ensemble_weights(self, X_test, y_test):
        # Проста рівна вага для всіх моделей
        n_models = len(self.models)
        if n_models == 0:
            self.ensemble_weights = {}
            return
        self.ensemble_weights = {name: 1.0 / n_models for name in self.models}

    def predict_proba(self, X: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        if model_name == 'ensemble':
            return self.predict_proba_ensemble(X)
        model = self.models.get(model_name)
        scaler = self.scalers.get('main')
        if model is None or scaler is None:
            raise ValueError("Model or scaler not found")
        X_scaled = scaler.transform(X)
        return model.predict_proba(X_scaled)[:, 1]

    def predict_proba_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Прогнозування ансамблем з урахуванням обробки даних"""
        # Обробка даних через feature processor
        if hasattr(self, 'feature_processor'):
            X_processed = self.feature_processor.transform(X)
            # Додавання interaction features
            X_processed = self.feature_extractor.create_interaction_features(X_processed)
        else:
            X_processed = X
        
        # Масштабування
        scaler = self.scalers.get('main')
        if scaler is None:
            raise ValueError("Scaler not found")
        X_scaled = scaler.transform(X_processed)
        
        # Прогнозування
        preds = np.zeros(X.shape[0])
        for model_name, model in self.models.items():
            weight = self.ensemble_weights.get(model_name, 1.0)
            preds += weight * model.predict_proba(X_scaled)[:, 1]
        
        # Нормалізація
        preds /= sum(self.ensemble_weights.values())
        
        return preds

    def predict_tender(self, tender_items: List[Dict], supplier_profiles: Dict) -> List[Dict]:
        features_list = []
        for item in tender_items:
            edrpou = item.get('EDRPOU')
            supplier_profile = supplier_profiles.get(edrpou)
            features = self.feature_extractor.extract_features(item, supplier_profile)
            features_list.append(features)
        X = pd.DataFrame(features_list).fillna(0)

        # Додаємо відсутні ознаки, якщо потрібно
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]  # впорядкувати колонки

        probs = self.predict_proba(X)
        results = []
        for i, item in enumerate(tender_items):
            results.append({
                'tender_number': item.get('F_TENDERNUMBER'),
                'edrpou': item.get('EDRPOU'),
                'probability': probs[i],
                'confidence': self._calculate_prediction_confidence(X.iloc[i], probs[i]),
                'risk_factors': self._identify_risk_factors(X.iloc[i], probs[i])
            })
            for i, result in enumerate(results):
                self.monitor.log_prediction(
                    tender_id=result['tender_number'],
                    features=features_list[i],  # Зберегти features_list
                    prediction=result['probability'],
                    model_version=f"ensemble_v{self.model_performance.get('version', '1.0')}"
                )
        return results

    def _calculate_prediction_confidence(self, features: pd.Series, probability: float) -> str:
        if probability > 0.8:
            return "high"
        elif probability > 0.6:
            return "medium"
        else:
            return "low"

    def _identify_risk_factors(self, features: pd.Series, probability: float) -> List[str]:
        risks = []
        if probability < 0.5:
            risks.append("Низька ймовірність перемоги")
        if features.get('competition_intensity', 0) > 0.7:
            risks.append("Висока конкуренція")
        if features.get('entry_barrier', 0) > 0.6:
            risks.append("Високий бар'єр входу")
        return risks

    def get_feature_analysis(self) -> Dict[str, pd.DataFrame]:
        return self.feature_importance

    def save_models(self, filepath: str):
        """Зберегти всі моделі, скейлери та ваги ансамблю у файл"""
        state = {
            'models': self.models,
            'scalers': self.scalers,
            'ensemble_weights': self.ensemble_weights,
            'feature_names': getattr(self, 'feature_names', []),
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        self.logger.info(f"Моделі збережено у {filepath}")

    def load_models(self, filepath: str):
        """Завантажити всі моделі, скейлери та ваги ансамблю з файлу"""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.models = state.get('models', {})
        self.scalers = state.get('scalers', {})
        self.ensemble_weights = state.get('ensemble_weights', {})
        self.feature_names = state.get('feature_names', [])
        self.feature_importance = state.get('feature_importance', {})
        self.model_performance = state.get('model_performance', {})
        self.logger.info(f"Моделі завантажено з {filepath}")