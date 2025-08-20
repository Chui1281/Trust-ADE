"""
Обертка для моделей scikit-learn
"""

import numpy as np
import warnings
from trust_ade.base_model import BaseModel
from trust_ade.utils import validate_inputs

class SklearnWrapper(BaseModel):
    """
    Обертка для моделей scikit-learn
    Обеспечивает единообразный интерфейс для всех sklearn моделей
    """

    def __init__(self, model, feature_names=None):
        """
        Инициализация wrapper

        Args:
            model: обученная sklearn модель
            feature_names: список имен признаков (опционально)
        """
        self.model = model

        # Определяем имена признаков
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            # Пытаемся определить автоматически
            if hasattr(model, 'n_features_in_'):
                n_features = model.n_features_in_
            elif hasattr(model, 'feature_importances_'):
                n_features = len(model.feature_importances_)
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                n_features = coef.shape[1] if coef.ndim > 1 else len(coef)
            else:
                n_features = 10  # Значение по умолчанию

            self.feature_names = [f"feature_{i}" for i in range(n_features)]

    def predict(self, X):
        """Предсказание классов"""
        X, _ = validate_inputs(X)
        try:
            return self.model.predict(X)
        except Exception as e:
            warnings.warn(f"Ошибка в predict: {str(e)}")
            return np.zeros(len(X))

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        X, _ = validate_inputs(X)

        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            elif hasattr(self.model, 'decision_function'):
                # Для моделей с decision_function (например, SVM)
                scores = self.model.decision_function(X)

                if scores.ndim == 1:
                    # Бинарная классификация
                    # Применяем сигмоиду для получения вероятностей
                    probs = 1 / (1 + np.exp(-scores))
                    return np.column_stack([1 - probs, probs])
                else:
                    # Многоклассовая классификация
                    # Применяем softmax
                    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            else:
                warnings.warn("Модель не поддерживает predict_proba или decision_function")
                # Возвращаем равномерные вероятности
                n_classes = len(np.unique(self.model.predict(X)))
                return np.full((len(X), n_classes), 1.0 / n_classes)

        except Exception as e:
            warnings.warn(f"Ошибка в predict_proba: {str(e)}")
            # Fallback - равномерные вероятности для бинарной классификации
            return np.full((len(X), 2), 0.5)

    def get_feature_names(self):
        """Получение имен признаков"""
        return self.feature_names.copy()

    def get_n_classes(self):
        """Получение количества классов"""
        if hasattr(self.model, 'classes_'):
            return len(self.model.classes_)
        else:
            return 2  # Предполагаем бинарную классификацию по умолчанию
