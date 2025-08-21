"""
Trust-ADE Bias Shift Index Module
Реализация BS_I = √(w_dp·DP_Δ² + w_eo·EO_Δ² + w_cf·CF_Δ²)
"""

import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from typing import Dict, List, Optional, Tuple, Any, Union


class BiasShiftIndex:
    """
    Класс для вычисления индекса смещения предвзятости согласно Trust-ADE

    Реализует формулы:
    - DP_Δ = |P(ŷ=1|A=0) - P(ŷ=1|A=1)|_t - |P(ŷ=1|A=0) - P(ŷ=1|A=1)|_t-Δt
    - EO_Δ = max_y |P(ŷ=1|A=0,y) - P(ŷ=1|A=1,y)|_t - max_y |P(ŷ=1|A=0,y) - P(ŷ=1|A=1,y)|_t-Δt
    - CF_Δ = |Acc(A=0) - Acc(A=1)|_t - |Acc(A=0) - Acc(A=1)|_t-Δt
    - BS_I = √(w_dp·DP_Δ² + w_eo·EO_Δ² + w_cf·CF_Δ²)
    """

    def __init__(self, protected_attributes: Optional[List[str]] = None,
                 dp_weight: float = 0.4, eo_weight: float = 0.4, cf_weight: float = 0.2,
                 min_group_size: int = 10):
        """
        Args:
            protected_attributes: список защищённых атрибутов для мониторинга
            dp_weight: вес демографического паритета
            eo_weight: вес равенства шансов
            cf_weight: вес калиброванной справедливости
            min_group_size: минимальный размер группы для анализа
        """
        self.protected_attributes = protected_attributes or []

        # Нормализация весов
        total = dp_weight + eo_weight + cf_weight
        self.w_dp = dp_weight / total
        self.w_eo = eo_weight / total
        self.w_cf = cf_weight / total

        self.min_group_size = max(1, min_group_size)

        print(f"⚖️ Trust-ADE Bias Shift Index initialized:")
        print(f"   w_dp (Demographic Parity): {self.w_dp:.3f}")
        print(f"   w_eo (Equalized Odds): {self.w_eo:.3f}")
        print(f"   w_cf (Calibrated Fairness): {self.w_cf:.3f}")
        print(f"   Min group size: {self.min_group_size}")

    def demographic_parity_shift(self, y_pred_current: np.ndarray,
                                 y_pred_baseline: np.ndarray,
                                 protected_attr: np.ndarray) -> float:
        """
        Trust-ADE формула демографического паритета:
        DP_Δ = |P(ŷ=1|A=0) - P(ŷ=1|A=1)|_t - |P(ŷ=1|A=0) - P(ŷ=1|A=1)|_t-Δt

        Args:
            y_pred_current: текущие предсказания [0/1]
            y_pred_baseline: базовые предсказания [0/1]
            protected_attr: значения защищённого атрибута

        Returns:
            float: изменение демографического паритета [0, 1]
        """
        try:
            protected_attr = np.array(protected_attr)
            y_pred_current = np.array(y_pred_current)
            y_pred_baseline = np.array(y_pred_baseline)

            # Проверка валидности данных
            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 0.0  # нет разделения на группы

            # Берём первые две группы для бинарного случая (A=0, A=1)
            group_0, group_1 = unique_groups[0], unique_groups[1]

            mask_0 = protected_attr == group_0
            mask_1 = protected_attr == group_1

            # Проверка минимального размера групп
            if np.sum(mask_0) < self.min_group_size or np.sum(mask_1) < self.min_group_size:
                return 0.0

            # P(ŷ=1|A=0) и P(ŷ=1|A=1) для текущего времени
            p_current_A0 = np.mean(y_pred_current[mask_0])
            p_current_A1 = np.mean(y_pred_current[mask_1])

            # P(ŷ=1|A=0) и P(ŷ=1|A=1) для базового времени
            p_baseline_A0 = np.mean(y_pred_baseline[mask_0])
            p_baseline_A1 = np.mean(y_pred_baseline[mask_1])

            # 📐 ФОРМУЛА TRUST-ADE
            current_disparity = abs(p_current_A0 - p_current_A1)
            baseline_disparity = abs(p_baseline_A0 - p_baseline_A1)

            dp_delta = abs(current_disparity - baseline_disparity)

            return np.clip(dp_delta, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в demographic_parity_shift: {str(e)}")
            return 0.5  # средний риск при ошибке

    def equalized_odds_shift(self, y_true: np.ndarray,
                             y_pred_current: np.ndarray,
                             y_pred_baseline: np.ndarray,
                             protected_attr: np.ndarray) -> float:
        """
        Trust-ADE формула равенства шансов:
        EO_Δ = max_y |P(ŷ=1|A=0,y) - P(ŷ=1|A=1,y)|_t - max_y |P(ŷ=1|A=0,y) - P(ŷ=1|A=1,y)|_t-Δt

        Args:
            y_true: истинные метки [0/1]
            y_pred_current: текущие предсказания [0/1]
            y_pred_baseline: базовые предсказания [0/1]
            protected_attr: значения защищённого атрибута

        Returns:
            float: изменение равенства шансов [0, 1]
        """
        try:
            y_true = np.array(y_true)
            protected_attr = np.array(protected_attr)
            y_pred_current = np.array(y_pred_current)
            y_pred_baseline = np.array(y_pred_baseline)

            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 0.0

            group_0, group_1 = unique_groups[0], unique_groups[1]

            mask_0 = protected_attr == group_0
            mask_1 = protected_attr == group_1

            if np.sum(mask_0) < self.min_group_size or np.sum(mask_1) < self.min_group_size:
                return 0.0

            # Вычисляем для y=0 и y=1 отдельно
            current_disparities = []
            baseline_disparities = []

            for y_class in [0, 1]:
                # Маски для y=y_class в каждой группе
                mask_0_y = mask_0 & (y_true == y_class)
                mask_1_y = mask_1 & (y_true == y_class)

                # Пропускаем если недостаточно примеров
                if np.sum(mask_0_y) < 5 or np.sum(mask_1_y) < 5:
                    continue

                # P(ŷ=1|A=0,y) и P(ŷ=1|A=1,y) для текущего времени
                p_current_A0_y = np.mean(y_pred_current[mask_0_y])
                p_current_A1_y = np.mean(y_pred_current[mask_1_y])

                # P(ŷ=1|A=0,y) и P(ŷ=1|A=1,y) для базового времени
                p_baseline_A0_y = np.mean(y_pred_baseline[mask_0_y])
                p_baseline_A1_y = np.mean(y_pred_baseline[mask_1_y])

                # Различия для каждого класса y
                current_disparities.append(abs(p_current_A0_y - p_current_A1_y))
                baseline_disparities.append(abs(p_baseline_A0_y - p_baseline_A1_y))

            if len(current_disparities) == 0:
                return 0.0

            # 📐 ФОРМУЛА TRUST-ADE - максимум по всем y
            max_current_disparity = max(current_disparities)
            max_baseline_disparity = max(baseline_disparities)

            eo_delta = abs(max_current_disparity - max_baseline_disparity)

            return np.clip(eo_delta, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в equalized_odds_shift: {str(e)}")
            return 0.5

    def calibrated_fairness_shift(self, y_true: np.ndarray,
                                  y_pred_current: np.ndarray,
                                  y_pred_baseline: np.ndarray,
                                  protected_attr: np.ndarray) -> float:
        """
        Trust-ADE формула калиброванной справедливости:
        CF_Δ = |Acc(A=0) - Acc(A=1)|_t - |Acc(A=0) - Acc(A=1)|_t-Δt

        Args:
            y_true: истинные метки [0/1]
            y_pred_current: текущие предсказания [0/1]
            y_pred_baseline: базовые предсказания [0/1]
            protected_attr: значения защищённого атрибута

        Returns:
            float: изменение калиброванной справедливости [0, 1]
        """
        try:
            y_true = np.array(y_true)
            protected_attr = np.array(protected_attr)
            y_pred_current = np.array(y_pred_current)
            y_pred_baseline = np.array(y_pred_baseline)

            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 0.0

            group_0, group_1 = unique_groups[0], unique_groups[1]

            mask_0 = protected_attr == group_0
            mask_1 = protected_attr == group_1

            if np.sum(mask_0) < self.min_group_size or np.sum(mask_1) < self.min_group_size:
                return 0.0

            # Accuracy для каждой группы в текущем времени
            try:
                acc_current_A0 = accuracy_score(y_true[mask_0], y_pred_current[mask_0])
                acc_current_A1 = accuracy_score(y_true[mask_1], y_pred_current[mask_1])

                # Accuracy для каждой группы в базовом времени
                acc_baseline_A0 = accuracy_score(y_true[mask_0], y_pred_baseline[mask_0])
                acc_baseline_A1 = accuracy_score(y_true[mask_1], y_pred_baseline[mask_1])

            except Exception as e:
                warnings.warn(f"🚨 Ошибка в вычислении accuracy: {str(e)}")
                return 0.5

            # 📐 ФОРМУЛА TRUST-ADE
            current_acc_disparity = abs(acc_current_A0 - acc_current_A1)
            baseline_acc_disparity = abs(acc_baseline_A0 - acc_baseline_A1)

            cf_delta = abs(current_acc_disparity - baseline_acc_disparity)

            return np.clip(cf_delta, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в calibrated_fairness_shift: {str(e)}")
            return 0.5

    def explanation_fairness_shift(self, explanations_current: Optional[np.ndarray],
                                   explanations_baseline: Optional[np.ndarray],
                                   protected_attr: np.ndarray) -> float:
        """
        Уникальная для Trust-ADE метрика: справедливость в объяснениях

        Измеряет различия в качестве объяснений между демографическими группами

        Args:
            explanations_current: текущие объяснения [n_samples, n_features]
            explanations_baseline: базовые объяснения [n_samples, n_features]
            protected_attr: значения защищённого атрибута

        Returns:
            float: изменение справедливости объяснений [0, 1]
        """
        try:
            if explanations_current is None or explanations_baseline is None:
                return 0.0

            explanations_current = np.array(explanations_current)
            explanations_baseline = np.array(explanations_baseline)
            protected_attr = np.array(protected_attr)

            unique_groups = np.unique(protected_attr)
            if len(unique_groups) < 2:
                return 0.0

            group_0, group_1 = unique_groups[0], unique_groups[1]

            mask_0 = protected_attr == group_0
            mask_1 = protected_attr == group_1

            if np.sum(mask_0) < self.min_group_size or np.sum(mask_1) < self.min_group_size:
                return 0.0

            # Средняя важность признаков для каждой группы
            current_importance_A0 = np.mean(np.abs(explanations_current[mask_0]), axis=0)
            current_importance_A1 = np.mean(np.abs(explanations_current[mask_1]), axis=0)

            baseline_importance_A0 = np.mean(np.abs(explanations_baseline[mask_0]), axis=0)
            baseline_importance_A1 = np.mean(np.abs(explanations_baseline[mask_1]), axis=0)

            # Косинусное расстояние между паттернами объяснений групп
            from scipy.spatial.distance import cosine

            current_explanation_disparity = cosine(current_importance_A0, current_importance_A1)
            baseline_explanation_disparity = cosine(baseline_importance_A0, baseline_importance_A1)

            # Обработка NaN значений
            if np.isnan(current_explanation_disparity):
                current_explanation_disparity = 0.5
            if np.isnan(baseline_explanation_disparity):
                baseline_explanation_disparity = 0.5

            explanation_fairness_delta = abs(current_explanation_disparity - baseline_explanation_disparity)

            return np.clip(explanation_fairness_delta, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в explanation_fairness_shift: {str(e)}")
            return 0.5

    def calculate_bias_shift_index(self, y_true: np.ndarray,
                                   y_pred_current: np.ndarray,
                                   y_pred_baseline: np.ndarray,
                                   protected_attr: np.ndarray,
                                   explanations_current: Optional[np.ndarray] = None,
                                   explanations_baseline: Optional[np.ndarray] = None) -> float:
        """
        Основная формула Trust-ADE для Bias Shift Index:
        BS_I = √(w_dp·DP_Δ² + w_eo·EO_Δ² + w_cf·CF_Δ²)

        Args:
            y_true: истинные метки
            y_pred_current: текущие предсказания
            y_pred_baseline: базовые предсказания
            protected_attr: значения защищённого атрибута
            explanations_current: текущие объяснения (опционально)
            explanations_baseline: базовые объяснения (опционально)

        Returns:
            float: Bias Shift Index [0, 1]
        """
        try:
            # Компоненты Trust-ADE
            dp_delta = self.demographic_parity_shift(y_pred_current, y_pred_baseline, protected_attr)
            eo_delta = self.equalized_odds_shift(y_true, y_pred_current, y_pred_baseline, protected_attr)
            cf_delta = self.calibrated_fairness_shift(y_true, y_pred_current, y_pred_baseline, protected_attr)

            # 🎯 ФОРМУЛА TRUST-ADE
            bs_index = np.sqrt(self.w_dp * dp_delta ** 2 +
                               self.w_eo * eo_delta ** 2 +
                               self.w_cf * cf_delta ** 2)

            return np.clip(bs_index, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в calculate_bias_shift_index: {str(e)}")
            return 0.5

    def calculate(self, y_true: Union[np.ndarray, list],
                  y_pred_current: Union[np.ndarray, list],
                  y_pred_baseline: Union[np.ndarray, list],
                  protected_data: Union[np.ndarray, list],
                  explanations_current: Optional[np.ndarray] = None,
                  explanations_baseline: Optional[np.ndarray] = None,
                  verbose: bool = True) -> Dict[str, float]:
        """
        Полный анализ Bias Shift согласно Trust-ADE с детальной разбивкой

        Args:
            y_true: истинные метки
            y_pred_current: текущие предсказания
            y_pred_baseline: базовые предсказания
            protected_data: данные о защищённых атрибутах
            explanations_current: текущие объяснения (опционально)
            explanations_baseline: базовые объяснения (опционально)
            verbose: детальный вывод результатов

        Returns:
            Dict[str, float]: детальные результаты анализа справедливости
        """
        try:
            if verbose:
                print(f"⚖️ Trust-ADE Bias Shift Analysis...")

            # Валидация и преобразование данных
            y_true = np.array(y_true).flatten()
            y_pred_current = np.array(y_pred_current).flatten()
            y_pred_baseline = np.array(y_pred_baseline).flatten()

            if protected_data is None or len(protected_data) == 0:
                if verbose:
                    print("⚠️  Нет данных о защищённых атрибутах - возвращаем нулевое смещение")
                return self._default_results()

            # Обработка защищённых атрибутов
            protected_attr = protected_data
            if isinstance(protected_data, (list, tuple)) and len(protected_data) > 0:
                protected_attr = protected_data[0]
            protected_attr = np.array(protected_attr).flatten()

            # Проверка размерностей
            if len(y_true) != len(y_pred_current) or len(y_true) != len(protected_attr):
                warnings.warn("🚨 Несовпадение размерностей данных")
                return self._default_results()

            # Компоненты смещения справедливости
            dp_delta = self.demographic_parity_shift(y_pred_current, y_pred_baseline, protected_attr)
            eo_delta = self.equalized_odds_shift(y_true, y_pred_current, y_pred_baseline, protected_attr)
            cf_delta = self.calibrated_fairness_shift(y_true, y_pred_current, y_pred_baseline, protected_attr)

            # Справедливость объяснений (уникально для Trust-ADE)
            explanation_fairness_delta = 0.0
            if explanations_current is not None and explanations_baseline is not None:
                explanation_fairness_delta = self.explanation_fairness_shift(
                    explanations_current, explanations_baseline, protected_attr
                )

            # 🎯 Основная формула Trust-ADE
            bias_shift_index = np.sqrt(self.w_dp * dp_delta ** 2 +
                                       self.w_eo * eo_delta ** 2 +
                                       self.w_cf * cf_delta ** 2)

            # Интерпретация уровня смещения
            if bias_shift_index < 0.1:
                bias_level = "Minimal"
            elif bias_shift_index < 0.3:
                bias_level = "Low"
            elif bias_shift_index < 0.5:
                bias_level = "Moderate"
            elif bias_shift_index < 0.7:
                bias_level = "High"
            else:
                bias_level = "Critical"

            results = {
                'bias_shift_index': np.clip(bias_shift_index, 0.0, 1.0),
                'demographic_parity_shift': np.clip(dp_delta, 0.0, 1.0),
                'equality_of_odds_shift': np.clip(eo_delta, 0.0, 1.0),
                'calibrated_fairness_shift': np.clip(cf_delta, 0.0, 1.0),
                'explanation_fairness_shift': np.clip(explanation_fairness_delta, 0.0, 1.0),
                'bias_level': bias_level,
                'protected_groups': len(np.unique(protected_attr)),
                'weights': {
                    'w_dp': self.w_dp,
                    'w_eo': self.w_eo,
                    'w_cf': self.w_cf
                }
            }

            if verbose:
                print(f"📊 Trust-ADE Bias Shift Results:")
                print(f"   🎯 Bias Shift Index: {results['bias_shift_index']:.4f} ({bias_level})")
                print(f"   📊 Demographic Parity Δ: {results['demographic_parity_shift']:.4f}")
                print(f"   ⚖️ Equalized Odds Δ: {results['equality_of_odds_shift']:.4f}")
                print(f"   📈 Calibrated Fairness Δ: {results['calibrated_fairness_shift']:.4f}")
                print(f"   🧠 Explanation Fairness Δ: {results['explanation_fairness_shift']:.4f}")
                print(f"   👥 Protected Groups: {results['protected_groups']}")

            return results

        except Exception as e:
            warnings.warn(f"🚨 Критическая ошибка в BiasShiftIndex.calculate: {str(e)}")
            return self._default_results()

    def _default_results(self) -> Dict[str, float]:
        """Результаты по умолчанию при критических ошибках"""
        return {
            'bias_shift_index': 0.0,
            'demographic_parity_shift': 0.0,
            'equality_of_odds_shift': 0.0,
            'calibrated_fairness_shift': 0.0,
            'explanation_fairness_shift': 0.0,
            'bias_level': 'Unknown',
            'protected_groups': 0,
            'weights': {
                'w_dp': self.w_dp,
                'w_eo': self.w_eo,
                'w_cf': self.w_cf
            }
        }