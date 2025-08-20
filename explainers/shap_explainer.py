"""
SHAP Explainer с правильной обработкой различных форматов вывода
"""

import numpy as np
import warnings
from trust_ade.utils import safe_explain
import shap

class SHAPExplainer:
    """
    Обертка для SHAP объяснений с правильной обработкой всех форматов вывода
    Основная цель - решить проблему "tuple index out of range"
    """

    def __init__(self, model, training_data=None, explainer_type='auto'):
        """
        Инициализация SHAP explainer

        Args:
            model: модель для объяснения
            training_data: обучающие данные для background
            explainer_type: тип SHAP explainer ('auto', 'tree', 'kernel')
        """
        self.model = model
        self.training_data = training_data
        self.explainer_type = explainer_type
        self.explainer = self._create_explainer()

    def _create_explainer(self):
        """Создание подходящего SHAP explainer"""
        try:
            import shap
        except ImportError:
            warnings.warn("SHAP не установлен. Используется fallback explainer.")
            return self._create_fallback_explainer()

        try:
            # Определяем тип модели и создаем соответствующий explainer
            model_obj = self.model.model if hasattr(self.model, 'model') else self.model

            # Tree-based модели
            if (self.explainer_type == 'tree' or
                    hasattr(model_obj, 'tree_') or
                    hasattr(model_obj, 'estimators_')):
                return shap.TreeExplainer(model_obj)

            # Kernel explainer для остальных моделей
            else:
                if self.training_data is not None:
                    # Ограничиваем размер background для производительности
                    sample_size = min(100, len(self.training_data))
                    background = shap.sample(self.training_data, sample_size)
                else:
                    # Создаем минимальный background
                    n_features = len(self.model.get_feature_names())
                    background = np.zeros((1, n_features))

                return shap.KernelExplainer(self.model.predict, background)

        except Exception as e:
            warnings.warn(f"Ошибка создания SHAP explainer: {str(e)}")
            return self._create_fallback_explainer()

    def _create_fallback_explainer(self):
        """Fallback explainer для случаев, когда SHAP недоступен"""
        return FallbackExplainer(self.model)

    def explain(self, X, class_idx=1, max_evals=100):
        """
        Генерация объяснений с правильной обработкой всех форматов

        Args:
            X: входные данные
            class_idx: индекс класса для многоклассовых задач
            max_evals: максимальное количество оценок для KernelExplainer

        Returns:
            numpy.ndarray: объяснения формы (n_samples, n_features)
        """
        return safe_explain(self, X, class_idx)

    def shap_values(self, X, **kwargs):
        """Прямой вызов shap_values для совместимости"""
        try:
            import shap
            if isinstance(self.explainer, shap.KernelExplainer):
                return self.explainer.shap_values(X, nsamples=kwargs.get('max_evals', 100))
            else:
                return self.explainer.shap_values(X)
        except Exception as e:
            warnings.warn(f"Ошибка в shap_values: {str(e)}")
            return self.explainer.explain(X)


class FallbackExplainer:
    """
    Fallback explainer для случаев, когда SHAP недоступен
    Использует важности признаков из модели
    """

    def __init__(self, model):
        self.model = model
        self.feature_importance = self._get_feature_importance()

    def _get_feature_importance(self):
        """Получение важности признаков из модели"""
        importance = self.model.get_feature_importance()

        if importance is not None:
            return importance

        # Если важности нет, используем равномерные веса
        n_features = len(self.model.get_feature_names())
        return np.ones(n_features) / n_features

    def explain(self, X):
        """Генерация простых объяснений на основе важности признаков"""
        try:
            if self.feature_importance is None:
                return np.random.random((len(X), X.shape[1])) * 0.1

            # Обеспечиваем правильную длину важностей
            if len(self.feature_importance) != X.shape[1]:
                importance = np.ones(X.shape[1]) / X.shape[1]
            else:
                importance = self.feature_importance

            # Генерируем объяснения как произведение входов на важность
            explanations = []
            for i in range(len(X)):
                explanation = X[i] * importance
                explanations.append(explanation)

            return np.array(explanations)

        except Exception as e:
            warnings.warn(f"Ошибка в fallback explainer: {str(e)}")
            return np.zeros((len(X), X.shape[1]))

    def shap_values(self, X):
        """Совместимость с SHAP интерфейсом"""
        return self.explain(X)
