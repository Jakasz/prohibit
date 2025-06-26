import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CategoryMapper:
    """Маппер для роботи з категоріями та кластерами"""
    
    def __init__(self, mapping_file: str = "data/categories_map.json"):
        self.mapping_file = Path(mapping_file)
        self.clusters = {}  # cluster_id -> list of categories
        self.category_to_cluster = {}  # category -> cluster_id
        self._load_mappings()
        
    def _load_mappings(self):
        """Завантажує маппінг категорій"""
        if not self.mapping_file.exists():
            logger.warning(f"Файл маппінгу {self.mapping_file} не знайдено")
            return
            
        try:
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                self.clusters = json.load(f)
                
            # Створюємо зворотній маппінг
            for cluster_id, categories in self.clusters.items():
                for category in categories:
                    # Нормалізуємо назви категорій
                    normalized = self._normalize_category_name(category)
                    self.category_to_cluster[normalized] = cluster_id
                    
            logger.info(f"✅ Завантажено {len(self.clusters)} кластерів з {len(self.category_to_cluster)} категоріями")
            
        except Exception as e:
            logger.error(f"Помилка завантаження маппінгу: {e}")
            
    def _normalize_category_name(self, category: str) -> str:
        """Нормалізує назву категорії"""
        if not category:
            return ""
        return category.lower().strip()
        
    def get_cluster_id(self, category_name: str) -> Optional[str]:
        """Повертає ID кластера для категорії"""
        normalized = self._normalize_category_name(category_name)
        return self.category_to_cluster.get(normalized)
        
    def get_cluster_categories(self, cluster_id: str) -> List[str]:
        """Повертає всі категорії в кластері"""
        return self.clusters.get(cluster_id, [])
        
    def get_related_categories(self, category_name: str) -> List[str]:
        """Повертає пов'язані категорії з того ж кластера"""
        cluster_id = self.get_cluster_id(category_name)
        if not cluster_id:
            return []
            
        # Повертаємо всі категорії крім поточної
        normalized_current = self._normalize_category_name(category_name)
        return [
            cat for cat in self.clusters.get(cluster_id, [])
            if self._normalize_category_name(cat) != normalized_current
        ]
        
    def get_all_clusters(self) -> Dict[str, List[str]]:
        """Повертає всі кластери"""
        return self.clusters.copy()
        
    def get_cluster_name(self, cluster_id: str) -> str:
        """Повертає читабельну назву кластера"""
        # Мапінг для українських назв
        cluster_names = {
            'agricultural_parts': 'Сільськогосподарські запчастини',
            'electronics': 'Електроніка та електротехніка',
            'construction': 'Будівельні матеріали',
            'office_supplies': 'Канцелярські товари',
            'medical': 'Медичні товари',
            'food': 'Продукти харчування',
            'fuel': 'Паливо та мастила',
            'services': 'Послуги',
            'communication': 'Зв\'язок та IT',
            'transport': 'Транспорт',
            'industrial': 'Промислове обладнання',
            'chemicals': 'Хімічна продукція'
        }
        
        return cluster_names.get(cluster_id, cluster_id.replace('_', ' ').title())
        
    def get_category_hierarchy(self) -> Dict[str, Dict]:
        """Повертає ієрархію категорій для UI"""
        hierarchy = {}
        
        for cluster_id, categories in self.clusters.items():
            hierarchy[cluster_id] = {
                'name': self.get_cluster_name(cluster_id),
                'categories': sorted(categories),
                'count': len(categories)
            }
            
        return hierarchy
        
    def find_similar_categories(self, category_name: str, limit: int = 5) -> List[str]:
        """Знаходить схожі категорії (навіть з інших кластерів)"""
        normalized = self._normalize_category_name(category_name)
        similar = []
        
        # Спочатку категорії з того ж кластера
        related = self.get_related_categories(category_name)
        similar.extend(related[:limit])
        
        # Якщо недостатньо, шукаємо в інших кластерах за схожістю назв
        if len(similar) < limit:
            for cat in self.category_to_cluster:
                if cat != normalized and cat not in similar:
                    # Проста перевірка на схожість
                    if any(word in cat for word in normalized.split() if len(word) > 3):
                        similar.append(cat)
                        if len(similar) >= limit:
                            break
                            
        return similar[:limit]
        
    def get_statistics(self) -> Dict[str, any]:
        """Повертає статистику по кластерах"""
        stats = {
            'total_clusters': len(self.clusters),
            'total_categories': len(self.category_to_cluster),
            'largest_cluster': None,
            'smallest_cluster': None,
            'avg_categories_per_cluster': 0
        }
        
        if self.clusters:
            cluster_sizes = {
                cluster_id: len(categories) 
                for cluster_id, categories in self.clusters.items()
            }
            
            stats['largest_cluster'] = max(cluster_sizes.items(), key=lambda x: x[1])
            stats['smallest_cluster'] = min(cluster_sizes.items(), key=lambda x: x[1])
            stats['avg_categories_per_cluster'] = sum(cluster_sizes.values()) / len(cluster_sizes)
            
        return stats