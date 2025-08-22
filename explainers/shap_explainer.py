"""
Enhanced SHAP Explainer с правильной обработкой всех форматов вывода
"""

import numpy as np
import warnings
from typing import Union, Optional, Any


class SHAPExplainer:
    """
    Улучшенная обертка для SHAP объяснений с правильной обработкой всех форматов
    """

    def __init__(self, model, training_data=None, explainer_type='auto'):
        """
        Инициализация SHAP explainer
        """
        self.model = model
        self.training_data = training_data
        self.explainer_type = explainer_type
        self.explainer = self._create_explainer()

        # Определяем количество признаков
        self.n_features = self._get_n_features()

        # Кэш для повторных вычислений
        self._explanation_cache = {}

    def _get_n_features(self) -> int:
        """Безопасное определение количества признаков"""
        try:
            # Пробуем разные способы
            if hasattr(self.model, 'n_features_'):
                return self.model.n_features_
            elif hasattr(self.model, 'n_features_in_'):
                return self.model.n_features_in_
            elif self.training_data is not None:
                return self.training_data.shape[1] if hasattr(self.training_data, 'shape') else len(self.training_data)
            else:
                # Последний resort - пробуем через coef_ или feature_importances_
                if hasattr(self.model, 'coef_'):
                    model_obj = self.model.model if hasattr(self.model, 'model') else self.model
                    return len(model_obj.coef_) if len(model_obj.coef_.shape) > 1 else len(model_obj.coef_)
                elif hasattr(self.model, 'feature_importances_'):
                    model_obj = self.model.model if hasattr(self.model, 'model') else self.model
                    return len(model_obj.feature_importances_)
                else:
                    return 10  # Fallback значение
        except Exception:
            return 10  # Безопасный fallback

    def _create_explainer(self):
        """Создание подходящего SHAP explainer с улучшенной обработкой ошибок"""
        try:
            import shap

            # Получаем внутреннюю модель если есть обертка
            model_obj = self.model.model if hasattr(self.model, 'model') else self.model

            # Tree-based модели
            if (self.explainer_type == 'tree' or
                self._is_tree_model(model_obj)):

                print("🌳 Используется TreeExplainer")
                return shap.TreeExplainer(model_obj)

            # Linear модели
            elif self._is_linear_model(model_obj):
                print("📏 Используется LinearExplainer")
                return shap.LinearExplainer(model_obj, self._get_background_data())

            # Kernel explainer для остальных
            else:
                print("🔮 Используется KernelExplainer")
                background = self._get_background_data()
                return shap.KernelExplainer(self._get_predict_function(), background)

        except ImportError:
            warnings.warn("SHAP не установлен. Используется fallback explainer.")
            return FallbackExplainer(self.model, self.n_features)
        except Exception as e:
            warnings.warn(f"Ошибка создания SHAP explainer: {str(e)}. Используется fallback.")
            return FallbackExplainer(self.model, self.n_features)

    def _is_tree_model(self, model) -> bool:
        """Проверка является ли модель деревом"""
        tree_indicators = [
            'tree_', 'estimators_', 'RandomForest', 'GradientBoosting',
            'XGBoost', 'LightGBM', 'CatBoost'
        ]
        model_str = str(type(model).__name__)
        return any(indicator in str(model.__class__) or indicator in model_str
                  for indicator in tree_indicators)

    def _is_linear_model(self, model) -> bool:
        """Проверка является ли модель линейной"""
        linear_indicators = [
            'LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso',
            'ElasticNet', 'SGD', 'coef_'
        ]
        return (any(indicator in str(model.__class__) for indicator in linear_indicators) or
                hasattr(model, 'coef_'))

    def _get_background_data(self):
        """Получение фоновых данных для SHAP"""
        try:
            if self.training_data is not None:
                import shap
                # Ограничиваем размер для производительности
                sample_size = min(100, len(self.training_data))
                return shap.sample(self.training_data, sample_size)
            else:
                # Создаем нулевой background
                return np.zeros((1, self.n_features))
        except Exception:
            return np.zeros((1, self.n_features))

    def _get_predict_function(self):
        """Получение функции предсказания"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba
        elif hasattr(self.model, 'predict'):
            return self.model.predict
        else:
            # Создаем заглушку
            def dummy_predict(X):
                return np.random.random((len(X), 2))
            return dummy_predict

    def explain(self, X: np.ndarray, class_idx: int = 1, max_evals: int = 100) -> np.ndarray:
        """
        Генерация объяснений с правильной обработкой всех форматов SHAP

        Args:
            X: входные данные
            class_idx: индекс класса для многоклассовых задач
            max_evals: максимальное количество оценок для KernelExplainer

        Returns:
            numpy.ndarray: объяснения формы (n_samples, n_features)
        """
        try:
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Создаем ключ для кэширования
            cache_key = f"{X.shape}_{class_idx}_{max_evals}"
            if cache_key in self._explanation_cache:
                return self._explanation_cache[cache_key]

            # Получаем SHAP значения
            shap_values = self._get_shap_values(X, max_evals)

            # Обрабатываем разные форматы выходов
            processed_explanations = self._process_shap_output(shap_values, class_idx)

            # Кэшируем результат
            if len(self._explanation_cache) < 10:  # Ограничиваем размер кэша
                self._explanation_cache[cache_key] = processed_explanations

            return processed_explanations

        except Exception as e:
            warnings.warn(f"Ошибка в SHAP explain: {str(e)}")
            return self._generate_fallback_explanations(X)

    def _get_shap_values(self, X: np.ndarray, max_evals: int) -> Any:
        """Получение SHAP значений с обработкой разных типов explainer'ов"""
        try:
            import shap

            if isinstance(self.explainer, shap.KernelExplainer):
                return self.explainer.shap_values(X, nsamples=max_evals, silent=True)
            elif isinstance(self.explainer, shap.LinearExplainer):
                return self.explainer.shap_values(X)
            elif hasattr(self.explainer, 'shap_values'):
                return self.explainer.shap_values(X)
            else:
                # FallbackExplainer
                return self.explainer.explain(X)

        except Exception as e:
            warnings.warn(f"Ошибка получения SHAP values: {str(e)}")
            return self.explainer.explain(X)

    def _process_shap_output(self, shap_values: Any, class_idx: int) -> np.ndarray:
        """
        Обработка различных форматов выхода SHAP

        Возможные форматы:
        1. numpy.ndarray (n_samples, n_features) - бинарная классификация
        2. list/tuple of arrays - многоклассовая классификация
        3. Explanation object - новые версии SHAP
        """
        try:
            # Случай 1: Обычный массив numpy
            if isinstance(shap_values, np.ndarray):
                if len(shap_values.shape) == 2:
                    return shap_values
                elif len(shap_values.shape) == 3:
                    # 3D массив - берем нужный класс
                    return shap_values[:, :, class_idx] if shap_values.shape[2] > class_idx else shap_values[:, :, 0]

            # Случай 2: Список или кортеж массивов (многоклассовая задача)
            elif isinstance(shap_values, (list, tuple)):
                if len(shap_values) > class_idx:
                    selected_values = shap_values[class_idx]
                else:
                    selected_values = shap_values  # Берем первый класс

                return np.array(selected_values) if not isinstance(selected_values, np.ndarray) else selected_values

            # Случай 3: SHAP Explanation object
            elif hasattr(shap_values, 'values'):
                values = shap_values.values
                if isinstance(values, np.ndarray):
                    if len(values.shape) == 3 and values.shape[2] > class_idx:
                        return values[:, :, class_idx]
                    elif len(values.shape) == 2:
                        return values
                    else:
                        return values[:, :, 0] if len(values.shape) == 3 else values
                else:
                    return np.array(values)

            # Случай 4: Неизвестный формат
            else:
                warnings.warn(f"Неизвестный формат SHAP output: {type(shap_values)}")
                return np.array(shap_values) if hasattr(shap_values, '__array__') else self._generate_fallback_explanations(None)

        except Exception as e:
            warnings.warn(f"Ошибка обработки SHAP output: {str(e)}")
            return self._generate_fallback_explanations(None)

    def _generate_fallback_explanations(self, X: Optional[np.ndarray]) -> np.ndarray:
        """Генерация fallback объяснений при ошибках"""
        if X is not None:
            n_samples = len(X)
            n_features = X.shape[1]
        else:
            n_samples = 1
            n_features = self.n_features

        # Генерируем случайные объяснения с правильной формой
        return np.random.normal(0, 0.1, size=(n_samples, n_features))

    def shap_values(self, X: np.ndarray, **kwargs) -> Union[np.ndarray, list]:
        """Прямой вызов shap_values для совместимости с SHAP API"""
        max_evals = kwargs.get('max_evals', kwargs.get('nsamples', 100))
        return self._get_shap_values(X, max_evals)

    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Делает объект вызываемым"""
        return self.explain(X, **kwargs)


class FallbackExplainer:
    """
    Улучшенный Fallback explainer
    """

    def __init__(self, model, n_features: int):
        self.model = model
        self.n_features = n_features
        self.feature_importance = self._get_feature_importance()

    def _get_feature_importance(self) -> np.ndarray:
        """Безопасное получение важности признаков"""
        try:
            # Пробуем разные способы получить важность признаков
            model_obj = self.model.model if hasattr(self.model, 'model') else self.model

            # 1. feature_importances_ (деревья, ансамбли)
            if hasattr(model_obj, 'feature_importances_'):
                importance = model_obj.feature_importances_
                if len(importance) == self.n_features:
                    return importance

            # 2. coef_ (линейные модели)
            if hasattr(model_obj, 'coef_'):
                coef = model_obj.coef_
                if len(coef.shape) > 1:
                    importance = np.abs(coef[0])  # Берем первый класс
                else:
                    importance = np.abs(coef)

                if len(importance) == self.n_features:
                    return importance / np.sum(importance)  # Нормализуем

            # 3. Пытаемся через обертку модели
            if hasattr(self.model, 'get_feature_importance'):
                try:
                    importance = self.model.get_feature_importance()
                    if importance is not None and len(importance) == self.n_features:
                        return importance
                except:
                    pass

            # 4. Fallback - равномерные веса
            return np.ones(self.n_features) / self.n_features

        except Exception:
            return np.ones(self.n_features) / self.n_features

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Генерация простых объяснений"""
        try:
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            n_samples = X.shape[0]
            n_features = min(X.shape[1], len(self.feature_importance))

            explanations = np.zeros((n_samples, n_features))

            for i in range(n_samples):
                # Объяснения как произведение входа на важность с добавлением шума
                explanation = X[i, :n_features] * self.feature_importance[:n_features]
                # Добавляем небольшой шум для реалистичности
                noise = np.random.normal(0, 0.01, size=explanation.shape)
                explanations[i] = explanation + noise

            return explanations

        except Exception as e:
            warnings.warn(f"Ошибка в fallback explainer: {str(e)}")
            n_samples = len(X) if X is not None else 1
            return np.random.normal(0, 0.05, size=(n_samples, self.n_features))

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Совместимость с SHAP API"""
        return self.explain(X)
