"""
Модули объяснения для различных типов моделей ML
"""

from shap_explainer import SHAPExplainer

__all__ = ['SHAPExplainer']

# Планируемые объяснители для будущих версий
PLANNED_EXPLAINERS = [
    'lime_explainer',
    'integrated_gradients_explainer',
    'counterfactual_explainer'
]
