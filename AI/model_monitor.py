# model_monitor.py
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from pathlib import Path

import pandas as pd


class ModelMonitor:
    """Моніторинг продуктивності моделей"""
    
    def __init__(self, log_dir: str = "./logs/predictions"):
        self.logger = logging.getLogger(__name__)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Логи прогнозів
        self.predictions_log = []
        self.performance_history = []
        
        # Метрики дрифту
        self.feature_distributions = {}
        self.prediction_distributions = []
        
        # Пороги для алертів
        self.drift_threshold = 0.05
        self.performance_drop_threshold = 0.1
        
    def log_prediction(self, 
                      tender_id: str, 
                      features: Dict,
                      prediction: float, 
                      model_version: str,
                      actual: Optional[bool] = None):
        """Логування прогнозу"""
        log_entry = {
            'tender_id': tender_id,
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'model_version': model_version,
            'actual': actual
        }
        
        self.predictions_log.append(log_entry)
        
        # Зберігання в файл кожні 100 записів
        if len(self.predictions_log) % 100 == 0:
            self._save_logs()
    
    def update_actual_outcome(self, tender_id: str, actual: bool):
        """Оновлення фактичного результату"""
        for log in reversed(self.predictions_log):
            if log['tender_id'] == tender_id:
                log['actual'] = actual
                break
    
    def calculate_drift(self, 
                       recent_features: pd.DataFrame,
                       reference_features: pd.DataFrame) -> Dict[str, float]:
        """Розрахунок дрифту ознак"""
        from scipy.stats import ks_2samp
        
        drift_scores = {}
        
        for column in recent_features.columns:
            if column in reference_features.columns:
                # Kolmogorov-Smirnov тест
                statistic, p_value = ks_2samp(
                    recent_features[column].dropna(),
                    reference_features[column].dropna()
                )
                drift_scores[column] = p_value
        
        # Загальний дрифт
        avg_drift = np.mean(list(drift_scores.values()))
        
        return {
            'feature_drift': drift_scores,
            'overall_drift': avg_drift,
            'drift_detected': avg_drift < self.drift_threshold
        }
    
    def calculate_performance_metrics(self, last_n: int = 100) -> Dict[str, float]:
        """Розрахунок метрик продуктивності"""
        # Фільтрація записів з фактичними результатами
        evaluated_predictions = [
            log for log in self.predictions_log[-last_n:]
            if log.get('actual') is not None
        ]
        
        if len(evaluated_predictions) < 10:
            return {'status': 'insufficient_data'}
        
        # Розрахунок метрик
        predictions = [log['prediction'] for log in evaluated_predictions]
        actuals = [log['actual'] for log in evaluated_predictions]
        
        # AUC
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        
        auc = roc_auc_score(actuals, predictions)
        
        # Precision/Recall при порозі 0.5
        binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
        precision = precision_score(actuals, binary_preds)
        recall = recall_score(actuals, binary_preds)
        
        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            'sample_size': len(evaluated_predictions)
        }
    
    def should_retrain(self) -> Tuple[bool, Dict[str, Any]]:
        """Визначення необхідності перенавчання"""
        reasons = []
        metrics = {}
        
        # 1. Перевірка дрифту (якщо достатньо даних)
        if len(self.predictions_log) >= 1000:
            recent_logs = self.predictions_log[-200:]
            historical_logs = self.predictions_log[-1000:-200]
            
            # Дрифт прогнозів
            recent_preds = [log['prediction'] for log in recent_logs]
            historical_preds = [log['prediction'] for log in historical_logs]
            
            from scipy.stats import ks_2samp
            _, p_value = ks_2samp(recent_preds, historical_preds)
            
            if p_value < self.drift_threshold:
                reasons.append(f"Prediction drift detected (p-value: {p_value:.4f})")
                metrics['prediction_drift'] = p_value
        
        # 2. Перевірка падіння продуктивності
        recent_performance = self.calculate_performance_metrics(100)
        historical_performance = self.calculate_performance_metrics(500)
        
        if ('auc' in recent_performance and 'auc' in historical_performance):
            perf_drop = historical_performance['auc'] - recent_performance['auc']
            
            if perf_drop > self.performance_drop_threshold:
                reasons.append(f"Performance drop: {perf_drop:.3f}")
                metrics['performance_drop'] = perf_drop
        
        # 3. Час з останнього тренування
        if hasattr(self, 'last_training_date'):
            days_since_training = (datetime.now() - self.last_training_date).days
            if days_since_training > 30:
                reasons.append(f"Model age: {days_since_training} days")
                metrics['model_age_days'] = days_since_training
        
        should_retrain = len(reasons) > 0
        
        return should_retrain, {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Генерація звіту моніторингу"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(self.predictions_log),
            'predictions_with_outcomes': sum(1 for log in self.predictions_log if log.get('actual') is not None)
        }
        
        # Продуктивність за останній період
        for period in [100, 500, 1000]:
            perf = self.calculate_performance_metrics(period)
            report[f'performance_last_{period}'] = perf
        
        # Перевірка необхідності перенавчання
        should_retrain, retrain_info = self.should_retrain()
        report['retraining_recommendation'] = retrain_info
        
        # Розподіл прогнозів
        if self.predictions_log:
            recent_preds = [log['prediction'] for log in self.predictions_log[-100:]]
            report['prediction_distribution'] = {
                'mean': np.mean(recent_preds),
                'std': np.std(recent_preds),
                'min': np.min(recent_preds),
                'max': np.max(recent_preds),
                'percentiles': {
                    '25': np.percentile(recent_preds, 25),
                    '50': np.percentile(recent_preds, 50),
                    '75': np.percentile(recent_preds, 75)
                }
            }
        
        return report
    
    def _save_logs(self):
        """Збереження логів у файл"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"predictions_{timestamp}.jsonl"
        
        with open(log_file, 'w') as f:
            for log in self.predictions_log[-1000:]:  # Зберігаємо останні 1000
                f.write(json.dumps(log) + '\n')