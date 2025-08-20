"""
Утилиты для Trust-ADE Protocol
Включает функцию safe_explain для правильной обработки SHAP вывода
"""

import numpy as np
import warnings


def safe_explain(explainer, X, class_idx=1, max_samples=100):
    """
    Безопасный вызов explainer с правильной обработкой всех форматов вывода

    Args:
        explainer: SHAP или другой explainer объект
        X: входные данные для объяснения
        class_idx: индекс класса (1 для положительного класса в бинарной классификации)
        max_samples: максимальное количество образцов для обработки

    Returns:
        numpy.ndarray: объяснения формы (n_samples, n_features)
    """
    try:
        # Ограничиваем количество образцов для производительности
        if len(X) > max_samples:
            X = X[:max_samples]

        # Различные способы получения объяснений
        result = None

        # Метод 1: Прямой SHAP explainer
        if hasattr(explainer, 'shap_values'):
            result = explainer.shap_values(X)
        # Метод 2: Wrapper с SHAP explainer внутри
        elif hasattr(explainer, 'explainer') and hasattr(explainer.explainer, 'shap_values'):
            result = explainer.explainer.shap_values(X)
        # Метод 3: Наш собственный explain метод
        elif hasattr(explainer, 'explain'):
            result = explainer.explain(X)
        else:
            warnings.warn("Explainer не поддерживает известные методы объяснения")
            return np.zeros((len(X), X.shape[1]))

        # Обработка различных форматов результата
        if isinstance(result, list):
            # Старый формат SHAP - список массивов для каждого класса
            if len(result) > class_idx:
                return np.array(result[class_idx])
            elif len(result) > 0:
                return np.array(result[0])
            else:
                return np.zeros((len(X), X.shape[1]))

        elif isinstance(result, np.ndarray):
            if result.ndim == 3:
                # Новый формат SHAP: (samples, features, classes)
                # Это основная причина ошибки "tuple index out of range"
                if result.shape[2] > class_idx:
                    return result[:, :, class_idx]
                else:
                    return result[:, :, 0]  # Fallback на первый класс
            elif result.ndim == 2:
                # Уже правильный формат: (samples, features)
                return result
            elif result.ndim == 1:
                # Одномерный массив - расширяем до 2D
                return result.reshape(1, -1)
            else:
                warnings.warn(f"Неожиданная размерность результата: {result.ndim}")
                return np.zeros((len(X), X.shape[1]))

        else:
            warnings.warn(f"Неожиданный тип результата: {type(result)}")
            return np.zeros((len(X), X.shape[1]))

    except Exception as e:
        warnings.warn(f"Ошибка в safe_explain: {str(e)}")
        return np.zeros((len(X), X.shape[1]))


def validate_inputs(X, y=None):
    """Валидация входных данных"""
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    if len(X.shape) != 2:
        raise ValueError(f"X должно быть 2D массивом, получено {len(X.shape)}D")

    if y is not None:
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if len(y) != len(X):
            raise ValueError(f"Длины X ({len(X)}) и y ({len(y)}) не совпадают")

    return X, y


def check_explainer_compatibility(explainer):
    """Проверка совместимости explainer"""
    has_shap = hasattr(explainer, 'shap_values')
    has_explainer_shap = hasattr(explainer, 'explainer') and hasattr(explainer.explainer, 'shap_values')
    has_explain = hasattr(explainer, 'explain')

    return has_shap or has_explainer_shap or has_explain


def ensure_2d_array(arr):
    """Обеспечивает, что массив является 2D"""
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    elif arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr
