"""
Модуль для вычисления индекса смещения предвзятости (Bias Shift Index)
"""

import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from .utils import validate_inputs


class BiasShiftIndex:
    """
    Класс для вычисления индекса смещения предвзятости
    Отслеживает изменения в справедливости модели между базовым и текущим состоянием
    """

    def __init__(self, protected_attributes=None):
        """
        Инициализация с защищенными атрибутами

        Args:
            protected_attributes: список защищенных атрибутов для мониторинга
        """
        self.protected_attributes = protected_attributes or []

    def demographic_parity_shift(self, y_pred_current, y_pred_baseline, protected_attr):
        """
        Изменение демографического паритета

        Args:
            y_pred_current: текущие предсказания
            y_pred_baseline: базовые предсказания
            protected_attr: значения защищенного атрибута

        Returns:
            float: изменение демографического паритета
        """
        try:
            if protected_attr is None or len(np.unique(protected_attr)) < 2:
                return 0.0

            groups = np.unique(protected_attr)
            current_rates = {}
            baseline_rates = {}

            for group in groups:
                mask = protected_attr == group
                if np.sum(mask) > 0:
                    current_rates[group] = np.mean(y_pred_current[mask])
                    baseline_rates[group] = np.mean(y_pred_baseline[mask])

            if len(current_rates) < 2:
                return 0.0

            # Максимальная разность между группами
            current_diff = max(current_rates.values()) - min(current_rates.values())
            baseline_diff = max(baseline_rates.values()) - min(baseline_rates.values())

            return abs(current_diff - baseline_diff)

        except Exception as e:
            warnings.warn(f"Ошибка в demographic_parity_shift: {str(e)}")
            return 0.0

    def equality_of_odds_shift(self, y_true, y_pred_current, y_pred_baseline, protected_attr):
        """
        Изменение равенства шансов

        Args:
            y_true: истинные метки
            y_pred_current: текущие предсказания
            y_pred_baseline: базовые предсказания
            protected_attr: значения защищенного атрибута

        Returns:
            float: изменение равенства шансов
        """
        try:
            if protected_attr is None or len(np.unique(protected_attr)) < 2:
                return 0.0

            groups = np.unique(protected_attr)
            current_tpr = {}
            baseline_tpr = {}

            for group in groups:
                mask = protected_attr == group
                if np.sum(mask) > 0:
                    group_y_true = y_true[mask]

                    # Проверяем, есть ли положительные примеры в группе
                    positive_mask = group_y_true == 1
                    if np.sum(positive_mask) > 0:
                        group_y_pred_curr = y_pred_current[mask]
                        group_y_pred_base = y_pred_baseline[mask]

                        # True Positive Rate для каждой группы
                        tp_curr = np.sum((group_y_true == 1) & (group_y_pred_curr == 1))
                        tp_base = np.sum((group_y_true == 1) & (group_y_pred_base == 1))
                        total_positive = np.sum(group_y_true == 1)

                        current_tpr[group] = tp_curr / total_positive
                        baseline_tpr[group] = tp_base / total_positive

            if len(current_tpr) < 2:
                return 0.0

            # Максимальная разность TPR между группами
            current_diff = max(current_tpr.values()) - min(current_tpr.values())
            baseline_diff = max(baseline_tpr.values()) - min(baseline_tpr.values())

            return abs(current_diff - baseline_diff)

        except Exception as e:
            warnings.warn(f"Ошибка в equality_of_odds_shift: {str(e)}")
            return 0.0

    def calibrated_fairness_shift(self, y_true, y_pred_current, y_pred_baseline, protected_attr):
        """
        Изменение калиброванной справедливости

        Args:
            y_true: истинные метки
            y_pred_current: текущие предсказания
            y_pred_baseline: базовые предсказания
            protected_attr: значения защищенного атрибута

        Returns:
            float: изменение калиброванной справедливости
        """
        try:
            if protected_attr is None or len(np.unique(protected_attr)) < 2:
                return 0.0

            groups = np.unique(protected_attr)
            current_acc = {}
            baseline_acc = {}

            for group in groups:
                mask = protected_attr == group
                if np.sum(mask) > 0:
                    try:
                        current_acc[group] = accuracy_score(y_true[mask], y_pred_current[mask])
                        baseline_acc[group] = accuracy_score(y_true[mask], y_pred_baseline[mask])
                    except Exception as e:
                        # Пропускаем группы с проблемами в вычислении accuracy
                        continue

            if len(current_acc) < 2:
                return 0.0

            # Максимальная разность точности между группами
            current_diff = max(current_acc.values()) - min(current_acc.values())
            baseline_diff = max(baseline_acc.values()) - min(baseline_acc.values())

            return abs(current_diff - baseline_diff)

        except Exception as e:
            warnings.warn(f"Ошибка в calibrated_fairness_shift: {str(e)}")
            return 0.0

    def calculate(self, y_true, y_pred_current, y_pred_baseline, protected_data,
                  dp_weight=0.4, eo_weight=0.4, cf_weight=0.2):
        """
        Вычисление итогового Bias Shift Index

        Args:
            y_true: истинные метки
            y_pred_current: текущие предсказания
            y_pred_baseline: базовые предсказания
            protected_data: данные о защищенных атрибутах
            dp_weight: вес демографического паритета
            eo_weight: вес равенства шансов
            cf_weight: вес калиброванной справедливости

        Returns:
            dict: словарь с результатами оценки
        """
        try:
            # Нормализуем веса
            total_weight = dp_weight + eo_weight + cf_weight
            dp_weight /= total_weight
            eo_weight /= total_weight
            cf_weight /= total_weight

            # Валидация входных данных
            y_true, _ = validate_inputs(np.array(y_true).reshape(-1, 1))
            y_true = y_true.flatten()

            y_pred_current = np.array(y_pred_current)
            y_pred_baseline = np.array(y_pred_baseline)

            if protected_data is None or len(protected_data) == 0:
                return {
                    'bias_shift_index': 0.0,
                    'demographic_parity_shift': 0.0,
                    'equality_of_odds_shift': 0.0,
                    'calibrated_fairness_shift': 0.0
                }

            # Берем первый защищенный атрибут если передан список
            protected_attr = protected_data
            if isinstance(protected_data, (list, tuple)) and len(protected_data) > 0:
                protected_attr = protected_data[0]

            protected_attr = np.array(protected_attr)
            
            # Вычисляем компоненты смещения
            dp_delta = self.demographic_parity_shift(y_pred_current, y_pred_baseline, protected_attr)
            eo_delta = self.equality_of_odds_shift(y_true, y_pred_current, y_pred_baseline, protected_attr)
            cf_delta = self.calibrated_fairness_shift(y_true, y_pred_current, y_pred_baseline, protected_attr)

            # Итоговый индекс смещения (использует L2 норму)
            bsi = np.sqrt(dp_weight * dp_delta ** 2 +
                          eo_weight * eo_delta ** 2 +
                          cf_weight * cf_delta ** 2)

            return {
                'bias_shift_index': max(0, min(1, bsi)),
                'demographic_parity_shift': max(0, min(1, dp_delta)),
                'equality_of_odds_shift': max(0, min(1, eo_delta)),
                'calibrated_fairness_shift': max(0, min(1, cf_delta))
            }

        except Exception as e:
            warnings.warn(f"Ошибка в BiasShiftIndex.calculate: {str(e)}")
            return {
                'bias_shift_index': 0.0,
                'demographic_parity_shift': 0.0,
                'equality_of_odds_shift': 0.0,
                'calibrated_fairness_shift': 0.0
            }
