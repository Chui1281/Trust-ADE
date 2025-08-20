"""
Trust-ADE Protocol - Trust Assessment through Dynamic Explainability

Протокол комплексной оценки доверия к системам ИИ через объединение
объяснимости, устойчивости и справедливости в единую метрику доверия.
"""

from .trust_ade import TrustADE
from .explainability_score import ExplainabilityScore
from .robustness_index import RobustnessIndex
from .bias_shift_index import BiasShiftIndex
from .concept_drift import ConceptDrift
from .trust_calculator import TrustCalculator
from .utils import safe_explain, validate_inputs, check_explainer_compatibility

__version__ = "1.0.0"
__author__ = "Trust-ADE Development Team"
__email__ = "trust-ade@example.com"

__all__ = [
    'TrustADE',
    'ExplainabilityScore',
    'RobustnessIndex',
    'BiasShiftIndex',
    'ConceptDrift',
    'TrustCalculator',
    'safe_explain',
    'validate_inputs',
    'check_explainer_compatibility'
]

# Версии компонентов для совместимости
COMPONENT_VERSIONS = {
    'explainability_score': '1.0.0',
    'robustness_index': '1.0.0',
    'bias_shift_index': '1.0.0',
    'concept_drift': '1.0.0',
    'trust_calculator': '1.0.0'
}

def get_version_info():
    """Получение информации о версии"""
    return {
        'version': __version__,
        'components': COMPONENT_VERSIONS,
        'python_required': '>=3.8'
    }
