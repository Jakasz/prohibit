# api_interface.py
from typing import Dict, List, Optional
import json
from datetime import datetime


class TenderPredictionAPI:
    """API інтерфейс для системи прогнозування"""
    
    def __init__(self, system):
        self.system = system
        
    def predict_single_tender(self, tender_data: Dict) -> Dict:
        """Прогноз для одного тендера"""
        # Валідація вхідних даних
        required_fields = ['F_TENDERNUMBER', 'EDRPOU', 'F_ITEMNAME', 'ITEM_BUDGET']
        for field in required_fields:
            if field not in tender_data:
                return {'error': f'Missing required field: {field}'}
        
        try:
            # Прогнозування
            results = self.system.predict_tender_outcomes(
                [tender_data],
                include_competition_analysis=True,
                include_similar_tenders=True
            )
            
            # Форматування відповіді
            tender_id = tender_data['F_TENDERNUMBER']
            prediction = results['predictions'].get(tender_id, {})
            competition = results['competition_analysis'].get(tender_id, {})
            similar = results['similar_tenders'].get(tender_id, [])[:5]
            
            return {
                'status': 'success',
                'tender_id': tender_id,
                'prediction': {
                    'win_probability': prediction.get('probability', 0),
                    'confidence': prediction.get('confidence', 'low'),
                    'risk_factors': prediction.get('risk_factors', [])
                },
                'competition': {
                    'level': competition.get('competition_forecast', {}).get('competition_level', 'unknown'),
                    'expected_participants': competition.get('competition_forecast', {}).get('expected_participants', 0),
                    'recommendations': competition.get('recommendations', [])
                },
                'similar_tenders': [
                    {
                        'tender_number': t.get('tender_number'),
                        'similarity': t.get('similarity_score'),
                        'won': t.get('won')
                    } for t in similar
                ],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_predict(self, tender_list: List[Dict]) -> Dict:
        """Прогноз для батчу тендерів"""
        results = []
        
        for tender in tender_list:
            result = self.predict_single_tender(tender)
            results.append(result)
        
        return {
            'status': 'success',
            'total_processed': len(tender_list),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_supplier_insights(self, edrpou: str) -> Dict:
        """Отримання інсайтів по постачальнику"""
        try:
            analytics = self.system.get_supplier_analytics(edrpou)
            
            if 'error' in analytics:
                return {
                    'status': 'error',
                    'error': analytics['error']
                }
            
            profile = analytics['profile']
            
            return {
                'status': 'success',
                'edrpou': edrpou,
                'supplier_name': profile.get('name', ''),
                'metrics': {
                    'total_tenders': profile.get('metrics', {}).get('total_tenders', 0),
                    'win_rate': profile.get('metrics', {}).get('win_rate', 0),
                    'reliability_score': profile.get('reliability_score', 0),
                    'market_position': profile.get('market_position', 'unknown')
                },
                'strengths': profile.get('competitive_advantages', []),
                'weaknesses': profile.get('weaknesses', []),
                'recommendations': analytics.get('recommendations', []),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }