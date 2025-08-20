"""
Модели и обертки для интеграции с различными ML библиотеками
"""

from .sklearn_wrapper import SklearnWrapper

__all__ = ['SklearnWrapper']

# Планируемые обертки для будущих версий
PLANNED_WRAPPERS = [
    'pytorch_wrapper',
    'tensorflow_wrapper',
    'xgboost_wrapper',
    'lightgbm_wrapper'
]
