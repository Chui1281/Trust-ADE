"""
Модуль для детектирования дрейфа концептов (Concept Drift Detection)
"""

import numpy as np
import warnings
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from .utils import validate_inputs


class ConceptDrift:
    """
    Класс для детектирования дрейфа концептов
    Отслеживает изменения в распределении данных и качестве объяснений
    """

    def __init__(self):
        """Инициализация детектора дрейфа концептов"""
        return

    def kolmogorov_smirnov_drift(self, X_reference, X_current):
        """
        KS тест для каждого признака для обнаружения дрейфа в данных

        Args:
            X_reference: референсные данные
            X_current: текущие данные

        Returns:
            float: средняя KS статистика по всем признакам
        """
        try:
            X_reference, _ = validate_inputs(X_reference)
            X_current, _ = validate_inputs(X_current)

            if X_reference.shape[1] != X_current.shape[1]:
                warnings.warn("Количество признаков в референсных и текущих данных не совпадает")
                return 1.0  # Максимальный дрейф

            ks_statistics = []

            for feature_idx in range(X_reference.shape[1]):
                ref_feature = X_reference[:, feature_idx]
                curr_feature = X_current[:, feature_idx]

                # Проверяем, что признаки не константы
                if np.var(ref_feature) < 1e-10 and np.var(curr_feature) < 1e-10:
                    ks_statistics.append(0.0)  # Нет дрейфа для константных признаков
                    continue

                try:
                    # KS тест
                    ks_stat, p_value = ks_2samp(ref_feature, curr_feature)
                    ks_statistics.append(ks_stat)
                except Exception as e:
                    warnings.warn(f"Ошибка KS теста для признака {feature_idx}: {str(e)}")
                    ks_statistics.append(0.5)  # Средний дрейф при ошибке

            return np.mean(ks_statistics) if ks_statistics else 0.0

        except Exception as e:
            warnings.warn(f"Ошибка в kolmogorov_smirnov_drift: {str(e)}")
            return 0.5

    def jensen_shannon_divergence(self, y_pred_reference, y_pred_current, n_bins=10):
        """
        Jensen-Shannon дивергенция для распределений предсказаний

        Args:
            y_pred_reference: референсные предсказания
            y_pred_current: текущие предсказания
            n_bins: количество бинов для дискретизации

        Returns:
            float: JS дивергенция
        """
        try:
            y_pred_reference = np.array(y_pred_reference)
            y_pred_current = np.array(y_pred_current)

            # Обрабатываем вероятности или дискретные предсказания
            if len(np.unique(y_pred_reference)) > n_bins:
                # Непрерывные предсказания - дискретизируем
                min_val = min(np.min(y_pred_reference), np.min(y_pred_current))
                max_val = max(np.max(y_pred_reference), np.max(y_pred_current))

                if max_val - min_val < 1e-10:
                    return 0.0  # Нет дрейфа для константных предсказаний

                bins = np.linspace(min_val, max_val, n_bins + 1)
                ref_hist, _ = np.histogram(y_pred_reference, bins=bins, density=True)
                curr_hist, _ = np.histogram(y_pred_current, bins=bins, density=True)
            else:
                # Дискретные предсказания
                unique_vals = np.unique(np.concatenate([y_pred_reference, y_pred_current]))
                ref_hist = np.array([np.sum(y_pred_reference == val) for val in unique_vals])
                curr_hist = np.array([np.sum(y_pred_current == val) for val in unique_vals])

            # Нормализация гистограмм
            ref_hist = ref_hist / (np.sum(ref_hist) + 1e-8)
            curr_hist = curr_hist / (np.sum(curr_hist) + 1e-8)

            # Добавляем малое значение для избежания логарифма от нуля
            ref_hist += 1e-8
            curr_hist += 1e-8

            # Вычисляем JS дивергенцию
            js_div = jensenshannon(ref_hist, curr_hist)

            # Обрабатываем NaN значения
            if np.isnan(js_div) or np.isinf(js_div):
                return 0.5

            return max(0, min(1, js_div))

        except Exception as e:
            warnings.warn(f"Ошибка в jensen_shannon_divergence: {str(e)}")
            return 0.5

    def explanation_drift(self, explanations_reference, explanations_current):
        """
        Дрейф в качестве объяснений

        Args:
            explanations_reference: референсные объяснения
            explanations_current: текущие объяснения

        Returns:
            float: мера дрейфа объяснений
        """
        try:
            if explanations_reference is None or explanations_current is None:
                return 0.0

            explanations_reference = np.array(explanations_reference)
            explanations_current = np.array(explanations_current)

            if len(explanations_reference) == 0 or len(explanations_current) == 0:
                return 0.0

            # Средние объяснения для каждого набора
            mean_ref_exp = np.mean(explanations_reference, axis=0)
            mean_curr_exp = np.mean(explanations_current, axis=0)

            # Проверяем размерности
            if len(mean_ref_exp) != len(mean_curr_exp):
                warnings.warn("Размерности референсных и текущих объяснений не совпадают")
                return 1.0

            # Косинусное расстояние между средними объяснениями
            from scipy.spatial.distance import cosine
            drift = cosine(mean_ref_exp, mean_curr_exp)

            # Обрабатываем NaN значения
            if np.isnan(drift) or np.isinf(drift):
                return 0.5

            return max(0, min(1, drift))

        except Exception as e:
            warnings.warn(f"Ошибка в explanation_drift: {str(e)}")
            return 0.5

    def calculate(self, X_reference, X_current, y_pred_reference, y_pred_current,
                  explanations_reference=None, explanations_current=None,
                  ks_weight=0.4, js_weight=0.4, exp_weight=0.2):
        """
        Вычисление итогового Concept Drift Rate

        Args:
            X_reference: референсные входные данные
            X_current: текущие входные данные
            y_pred_reference: референсные предсказания
            y_pred_current: текущие предсказания
            explanations_reference: референсные объяснения (опционально)
            explanations_current: текущие объяснения (опционально)
            ks_weight: вес KS дрейфа
            js_weight: вес JS дивергенции
            exp_weight: вес дрейфа объяснений

        Returns:
            dict: словарь с результатами оценки дрейфа
        """
        try:
            # Нормализуем веса
            total_weight = ks_weight + js_weight + exp_weight
            ks_weight /= total_weight
            js_weight /= total_weight
            exp_weight /= total_weight

            # KS дрейф данных
            ks_drift = self.kolmogorov_smirnov_drift(X_reference, X_current)

            # JS дивергенция предсказаний
            js_drift = self.jensen_shannon_divergence(y_pred_reference, y_pred_current)

            # Дрейф объяснений
            exp_drift = 0.0
            if explanations_reference is not None and explanations_current is not None:
                exp_drift = self.explanation_drift(explanations_reference, explanations_current)

            # Итоговый дрейф концептов
            cdr = (ks_weight * ks_drift +
                   js_weight * js_drift +
                   exp_weight * exp_drift)

            return {
                'concept_drift_rate': max(0, min(1, cdr)),
                'ks_drift': max(0, min(1, ks_drift)),
                'js_divergence': max(0, min(1, js_drift)),
                'explanation_drift': max(0, min(1, exp_drift))
            }

        except Exception as e:
            warnings.warn(f"Ошибка в ConceptDrift.calculate: {str(e)}")
            return {
                'concept_drift_rate': 0.0,
                'ks_drift': 0.0,
                'js_divergence': 0.0,
                'explanation_drift': 0.0
            }
