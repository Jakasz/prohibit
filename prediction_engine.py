import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Tuple, Optional, Any
import pickle
import joblib

class PredictionEngine:
    """Основний двигун прогнозування тендерів"""

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
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

        self.ensemble_weights = {}
        self.calibrated_models = {}

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

    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, use_calibration: bool = True):
        self.logger.info("Початок тренування моделей...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['main'] = scaler

        for model_name, config in self.model_configs.items():
            model = config['class'](**config['params'])
            if use_calibration:
                model = CalibratedClassifierCV(model, cv=3)
            model.fit(X_train_scaled, y_train)
            self.models[model_name] = model
            perf = self._evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name)
            self.model_performance[model_name] = perf
            self._analyze_feature_importance(model, self.feature_names, model_name)

        self._optimize_ensemble_weights(X_test_scaled, y_test)
        ensemble_pred = self.predict_proba_ensemble(X_test)
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        self.logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
        self.model_performance['ensemble'] = {'test_auc': ensemble_auc}
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
        scaler = self.scalers.get('main')
        if scaler is None:
            raise ValueError("Scaler not found")
        X_scaled = scaler.transform(X)
        preds = np.zeros(X.shape[0])
        for model_name, model in self.models.items():
            weight = self.ensemble_weights.get(model_name, 1.0)
            preds += weight * model.predict_proba(X_scaled)[:, 1]
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