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

from supplier_profiler import SupplierProfile

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
from feature_extractor import FeatureExtractor
from feature_processor import AdvancedFeatureProcessor
from model_monitor import ModelMonitor

class PredictionEngine:
    """Основний двигун прогнозування тендерів"""

    def __init__(self, supplier_profiler, competition_analyzer, categories_manager):
        self.supplier_profiler = supplier_profiler
        self.competition_analyzer = competition_analyzer
        self.categories_manager = categories_manager
        self.feature_extractor = FeatureExtractor(categories_manager, competition_analyzer)
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
                    'min_samples_split': 10,
                    'min_samples_leaf': 5,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                    'random_state': 42,
                    'class_weight': 'balanced'
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'random_state': 42
                }
            }
            # XGBoost можна додати за потреби
        }
        if XGBOOST_AVAILABLE:
            self.model_configs['xgboost'] = {
                'class': XGBClassifier,
                'params': {
                    'n_estimators': 150,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss'
                }
            }


        self.ensemble_weights = {}
        self.calibrated_models = {}

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
        X_train_scaled = scaler.fit_transform(X_train_processed)  # <- змінено
        X_test_scaled = scaler.transform(X_test_processed)        # <- змінено
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
        # Замість: self._optimize_ensemble_weights(X_test_scaled, y_test)
        # Використовуємо:
        self._optimize_ensemble_weights_advanced(X_test_processed, y_test)  # <- НОВЕ
        
        # Оцінка ансамблю
        ensemble_pred = self.predict_proba_ensemble(X_test_processed)  # <- змінено
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        self.logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
        self.model_performance['ensemble'] = {'test_auc': ensemble_auc}
        
        # ===== НОВЕ: Збереження додаткової інформації =====
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