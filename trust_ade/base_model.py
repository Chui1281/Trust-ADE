"""
Базовый абстрактный класс для всех моделей ML в Trust-ADE Protocol
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Базовый абстрактный класс для всех моделей ML
    Определяет интерфейс, который должны реализовать все модели
    """

    @abstractmethod
    def predict(self, X):
        """
        Предсказание классов

        Args:
            X: входные данные

        Returns:
            numpy.ndarray: предсказанные классы
        """
        return np.array([])

    @abstractmethod
    def predict_proba(self, X):
        """
        Предсказание вероятностей принадлежности к классам

        Args:
            X: входные данные

        Returns:
            numpy.ndarray: матрица вероятностей
        """
        return np.array([[]])

    @abstractmethod
    def get_feature_names(self):
        """
        Получение имен признаков

        Returns:
            list: список имен признаков
        """
        return []

    def get_feature_importance(self):
        """
        Получение важности признаков (если доступно)

        Returns:
            numpy.ndarray: важности признаков или None
        """
        # Пытаемся получить важности из разных атрибутов
        if hasattr(self, 'model'):
            model = self.model
        else:
            model = self

        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                return np.mean(np.abs(coef), axis=0)
            else:
                return np.abs(coef)
        else:
            return None

    def get_n_features(self):
        """Получение количества признаков"""
        feature_names = self.get_feature_names()
        if feature_names:
            return len(feature_names)

        importance = self.get_feature_importance()
        if importance is not None:
            return len(importance)

        return None
