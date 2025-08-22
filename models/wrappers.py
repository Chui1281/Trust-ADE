"""
Обертки для моделей и заглушки
"""
import numpy as np
from sklearn.metrics import accuracy_score
from .sklearn_wrapper import SklearnWrapper
from trust_ade.trust_ade import TrustADE

class CUDAMLPWrapper(SklearnWrapper):
    """Обертка для оптимизированной CUDA MLP"""

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(10)]

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_names(self):
        return self.feature_names


class FixedXANFISWrapper(SklearnWrapper):
    """Исправленная обертка для XANFIS без неподдерживаемых параметров"""

    def __init__(self, model, feature_names=None, scaler=None):
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(10)]
        self.scaler = scaler
        self._is_fitted = True

    def predict(self, X):
        """Предсказание с улучшенной обработкой ошибок"""
        try:
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            pred = self.model.predict(X_scaled)

            # Нормализация результата
            if hasattr(pred, 'ravel'):
                pred = pred.ravel()

            # Обеспечиваем правильный тип
            pred = np.asarray(pred, dtype=int)

            # Проверяем размерность и валидность
            if len(pred) != len(X):
                print(f"⚠️ Размер предсказаний не совпадает: {len(pred)} vs {len(X)}")
                return np.zeros(len(X), dtype=int)

            # Проверяем валидность классов
            unique_pred = np.unique(pred)
            if len(unique_pred) == 0 or np.any(pred < 0):
                print(f"⚠️ Некорректные предсказания XANFIS")
                return np.zeros(len(X), dtype=int)

            return pred

        except Exception as e:
            print(f"❌ Ошибка в XANFIS predict: {str(e)}")
            return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        """Улучшенное предсказание вероятностей"""
        try:
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)

                if proba.ndim == 1:
                    # Бинарная классификация
                    proba_binary = np.column_stack([1 - proba, proba])
                    return proba_binary
                elif proba.ndim == 2:
                    return proba
                else:
                    raise ValueError(f"Неожиданная размерность: {proba.ndim}")
            else:
                # Создаем вероятности на основе предсказаний
                pred = self.predict(X)
                n_classes = len(np.unique(pred)) if len(np.unique(pred)) > 1 else 2

                proba = np.zeros((len(X), n_classes))
                for i, p in enumerate(pred):
                    if 0 <= p < n_classes:
                        proba[i, p] = 0.8
                        remaining = 0.2 / (n_classes - 1) if n_classes > 1 else 0
                        proba[i, :] += remaining
                        proba[i, p] = 0.8
                    else:
                        proba[i, :] = 1.0 / n_classes

                return proba

        except Exception as e:
            print(f"❌ Ошибка в XANFIS predict_proba: {str(e)}")
            return np.full((len(X), 2), 0.5)

    def get_feature_names(self):
        return self.feature_names
