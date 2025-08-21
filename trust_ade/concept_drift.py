"""
Trust-ADE Concept Drift Detection Module
Реализация CD_R = λ·KS(P_t, P_t-Δt) + (1-λ)·JS(P_t, P_t-Δt)
"""

import numpy as np
import warnings
from scipy.stats import ks_2samp, entropy
from scipy.spatial.distance import jensenshannon, cosine
from typing import Dict, List, Optional, Tuple, Any, Union


class ConceptDrift:
    """
    Класс для детектирования дрейфа концептов согласно Trust-ADE

    Реализует формулу: CD_R = λ·KS(P_t, P_t-Δt) + (1-λ)·JS(P_t, P_t-Δt)
    где:
    - KS: статистика Колмогорова-Смирнова
    - JS: дивергенция Йенсена-Шеннона
    - λ: параметр балансировки между KS и JS
    """

    def __init__(self, lambda_param: float = 0.5, n_bins: int = 10,
                 significance_level: float = 0.05):
        """
        Args:
            lambda_param: λ - параметр балансировки KS и JS (0 ≤ λ ≤ 1)
            n_bins: количество бинов для дискретизации непрерывных данных
            significance_level: уровень значимости для статистических тестов
        """
        self.lambda_param = np.clip(lambda_param, 0.0, 1.0)
        self.n_bins = max(5, n_bins)  # минимум 5 бинов
        self.significance_level = significance_level

        print(f"🔄 Trust-ADE Concept Drift Detector initialized:")
        print(f"   λ (KS weight): {self.lambda_param:.3f}")
        print(f"   JS weight: {1 - self.lambda_param:.3f}")
        print(f"   Bins: {self.n_bins}")

    def kolmogorov_smirnov_drift(self, X_reference: np.ndarray,
                                 X_current: np.ndarray) -> float:
        """
        Статистика Колмогорова-Смирнова для обнаружения дрейфа данных

        KS тест сравнивает эмпирические функции распределения двух выборок

        Args:
            X_reference: референсные данные [n_samples, n_features]
            X_current: текущие данные [n_samples, n_features]

        Returns:
            float: средняя KS статистика по всем признакам [0, 1]
        """
        try:
            X_reference = np.array(X_reference)
            X_current = np.array(X_current)

            # Валидация размерностей
            if len(X_reference.shape) == 1:
                X_reference = X_reference.reshape(-1, 1)
            if len(X_current.shape) == 1:
                X_current = X_current.reshape(-1, 1)

            if X_reference.shape[1] != X_current.shape[1]:
                warnings.warn("🚨 Количество признаков не совпадает")
                return 1.0  # максимальный дрейф

            ks_statistics = []

            for feature_idx in range(X_reference.shape[1]):
                ref_feature = X_reference[:, feature_idx]
                curr_feature = X_current[:, feature_idx]

                # Проверка на константные признаки
                ref_var = np.var(ref_feature)
                curr_var = np.var(curr_feature)

                if ref_var < 1e-12 and curr_var < 1e-12:
                    # Оба признака константны
                    if np.abs(np.mean(ref_feature) - np.mean(curr_feature)) < 1e-12:
                        ks_statistics.append(0.0)  # идентичные константы
                    else:
                        ks_statistics.append(1.0)  # разные константы
                    continue

                try:
                    # Двухвыборочный KS тест
                    ks_statistic, p_value = ks_2samp(ref_feature, curr_feature)

                    # KS статистика уже нормализована в [0, 1]
                    ks_statistics.append(np.clip(ks_statistic, 0.0, 1.0))

                except Exception as e:
                    warnings.warn(f"🚨 KS тест failed для признака {feature_idx}: {str(e)}")
                    ks_statistics.append(0.5)  # средний дрейф при ошибке

            # Средняя KS статистика по всем признакам
            mean_ks = np.mean(ks_statistics) if ks_statistics else 0.0
            return np.clip(mean_ks, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Критическая ошибка в kolmogorov_smirnov_drift: {str(e)}")
            return 0.5

    def jensen_shannon_divergence(self, P_reference: np.ndarray,
                                  P_current: np.ndarray) -> float:
        """
        Дивергенция Йенсена-Шеннона согласно формуле Trust-ADE

        JS(P, Q) = 0.5 * [KL(P || M) + KL(Q || M)], где M = 0.5 * (P + Q)

        Args:
            P_reference: референсное распределение (предсказания или данные)
            P_current: текущее распределение

        Returns:
            float: JS дивергенция [0, 1]
        """
        try:
            P_reference = np.array(P_reference).flatten()
            P_current = np.array(P_current).flatten()

            if len(P_reference) == 0 or len(P_current) == 0:
                return 0.0

            # Определяем тип данных (дискретные или непрерывные)
            ref_unique = len(np.unique(P_reference))
            curr_unique = len(np.unique(P_current))

            if ref_unique <= self.n_bins and curr_unique <= self.n_bins:
                # Дискретные данные - прямое вычисления распределения
                all_values = np.unique(np.concatenate([P_reference, P_current]))

                ref_counts = np.array([np.sum(P_reference == val) for val in all_values])
                curr_counts = np.array([np.sum(P_current == val) for val in all_values])

            else:
                # Непрерывные данные - дискретизация через гистограммы
                min_val = min(np.min(P_reference), np.min(P_current))
                max_val = max(np.max(P_reference), np.max(P_current))

                if max_val - min_val < 1e-12:
                    return 0.0  # идентичные константы

                # Создаем единые бины для обеих выборок
                bins = np.linspace(min_val, max_val, self.n_bins + 1)

                ref_counts, _ = np.histogram(P_reference, bins=bins)
                curr_counts, _ = np.histogram(P_current, bins=bins)

            # Нормализация в вероятностные распределения
            ref_probs = ref_counts / (np.sum(ref_counts) + 1e-12)
            curr_probs = curr_counts / (np.sum(curr_counts) + 1e-12)

            # Сглаживание для избежания логарифма от нуля
            epsilon = 1e-12
            ref_probs = ref_probs + epsilon
            curr_probs = curr_probs + epsilon

            # Ренормализация после сглаживания
            ref_probs = ref_probs / np.sum(ref_probs)
            curr_probs = curr_probs / np.sum(curr_probs)

            # Jensen-Shannon дивергенция
            js_divergence = jensenshannon(ref_probs, curr_probs)

            # Обработка NaN/Inf значений
            if np.isnan(js_divergence) or np.isinf(js_divergence):
                return 0.5

            # JS дивергенция уже в диапазоне [0, 1]
            return np.clip(js_divergence, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в jensen_shannon_divergence: {str(e)}")
            return 0.5

    def explanation_quality_drift(self, explanations_reference: np.ndarray,
                                  explanations_current: np.ndarray) -> float:
        """
        Дрейф качества объяснений - уникальная особенность Trust-ADE

        Отслеживает деградацию объяснимости модели во времени

        Args:
            explanations_reference: референсные объяснения [n_samples, n_features]
            explanations_current: текущие объяснения [n_samples, n_features]

        Returns:
            float: мера дрейфа объяснений [0, 1]
        """
        try:
            if explanations_reference is None or explanations_current is None:
                return 0.0

            explanations_reference = np.array(explanations_reference)
            explanations_current = np.array(explanations_current)

            if explanations_reference.size == 0 or explanations_current.size == 0:
                return 0.0

            # Приводим к 2D массивам
            if len(explanations_reference.shape) == 1:
                explanations_reference = explanations_reference.reshape(-1, 1)
            if len(explanations_current.shape) == 1:
                explanations_current = explanations_current.reshape(-1, 1)

            # Проверка совместимости размерностей
            if explanations_reference.shape[1] != explanations_current.shape[1]:
                warnings.warn("🚨 Размерности объяснений не совпадают")
                return 1.0

            # Метрики дрейфа объяснений
            drift_metrics = []

            # 1. Косинусное расстояние между средними объяснениями
            mean_ref = np.mean(explanations_reference, axis=0)
            mean_curr = np.mean(explanations_current, axis=0)

            if np.linalg.norm(mean_ref) > 1e-12 and np.linalg.norm(mean_curr) > 1e-12:
                cosine_drift = cosine(mean_ref, mean_curr)
                if not (np.isnan(cosine_drift) or np.isinf(cosine_drift)):
                    drift_metrics.append(cosine_drift)

            # 2. Разность энтропий распределений важности признаков
            ref_importance = np.abs(explanations_reference).mean(axis=0)
            curr_importance = np.abs(explanations_current).mean(axis=0)

            # Нормализация в вероятностные распределения
            ref_importance = ref_importance / (np.sum(ref_importance) + 1e-12)
            curr_importance = curr_importance / (np.sum(curr_importance) + 1e-12)

            ref_entropy = entropy(ref_importance + 1e-12)
            curr_entropy = entropy(curr_importance + 1e-12)

            max_entropy = np.log(len(ref_importance))
            if max_entropy > 1e-12:
                entropy_drift = abs(ref_entropy - curr_entropy) / max_entropy
                drift_metrics.append(entropy_drift)

            # 3. KS тест для распределений объяснений по признакам
            ks_drifts = []
            for feature_idx in range(explanations_reference.shape[1]):
                ref_feature_exp = explanations_reference[:, feature_idx]
                curr_feature_exp = explanations_current[:, feature_idx]

                if np.var(ref_feature_exp) > 1e-12 or np.var(curr_feature_exp) > 1e-12:
                    try:
                        ks_stat, _ = ks_2samp(ref_feature_exp, curr_feature_exp)
                        ks_drifts.append(ks_stat)
                    except:
                        continue

            if ks_drifts:
                drift_metrics.append(np.mean(ks_drifts))

            # Агрегированная метрика дрейфа объяснений
            if drift_metrics:
                explanation_drift = np.mean(drift_metrics)
                return np.clip(explanation_drift, 0.0, 1.0)
            else:
                return 0.5

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в explanation_quality_drift: {str(e)}")
            return 0.5

    def calculate_concept_drift_rate(self, X_reference: np.ndarray,
                                     X_current: np.ndarray,
                                     y_pred_reference: Optional[np.ndarray] = None,
                                     y_pred_current: Optional[np.ndarray] = None,
                                     explanations_reference: Optional[np.ndarray] = None,
                                     explanations_current: Optional[np.ndarray] = None) -> float:
        """
        Основная формула Trust-ADE для Concept-Drift Rate:
        CD_R = λ·KS(P_t, P_t-Δt) + (1-λ)·JS(P_t, P_t-Δt)

        Args:
            X_reference: референсные входные данные
            X_current: текущие входные данные
            y_pred_reference: референсные предсказания (опционально)
            y_pred_current: текущие предсказания (опционально)
            explanations_reference: референсные объяснения (опционально)
            explanations_current: текущие объяснения (опционально)

        Returns:
            float: Concept-Drift Rate [0, 1]
        """
        try:
            # 1. KS компонента - дрейф входных данных
            ks_component = self.kolmogorov_smirnov_drift(X_reference, X_current)

            # 2. JS компонента - выбираем наилучший источник данных
            js_component = 0.0

            if y_pred_reference is not None and y_pred_current is not None:
                # Предпочитаем предсказания как более информативные
                js_component = self.jensen_shannon_divergence(y_pred_reference, y_pred_current)
            else:
                # Fallback к анализу входных данных
                # Используем первый компонент PCA или среднее по признакам
                ref_summary = np.mean(X_reference, axis=1)
                curr_summary = np.mean(X_current, axis=1)
                js_component = self.jensen_shannon_divergence(ref_summary, curr_summary)

            # 3. 🎯 ФОРМУЛА TRUST-ADE
            cd_rate = (self.lambda_param * ks_component +
                       (1 - self.lambda_param) * js_component)

            return np.clip(cd_rate, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в calculate_concept_drift_rate: {str(e)}")
            return 0.5

    def calculate(self, X_reference: np.ndarray, X_current: np.ndarray,
                  y_pred_reference: Optional[np.ndarray] = None,
                  y_pred_current: Optional[np.ndarray] = None,
                  explanations_reference: Optional[np.ndarray] = None,
                  explanations_current: Optional[np.ndarray] = None,
                  verbose: bool = True) -> Dict[str, float]:
        """
        Полный анализ Concept Drift согласно Trust-ADE с детальной разбивкой

        Args:
            X_reference: референсные входные данные
            X_current: текущие входные данные
            y_pred_reference: референсные предсказания
            y_pred_current: текущие предсказания
            explanations_reference: референсные объяснения
            explanations_current: текущие объяснения
            verbose: детальный вывод результатов

        Returns:
            Dict[str, float]: детальные результаты анализа дрейфа
        """
        try:
            if verbose:
                print(f"🔄 Trust-ADE Concept Drift Analysis...")

            # Компоненты дрейфа
            ks_drift = self.kolmogorov_smirnov_drift(X_reference, X_current)

            # JS дивергенция предсказаний или данных
            js_drift = 0.0
            if y_pred_reference is not None and y_pred_current is not None:
                js_drift = self.jensen_shannon_divergence(y_pred_reference, y_pred_current)
                data_source = "predictions"
            else:
                ref_summary = np.mean(X_reference, axis=1)
                curr_summary = np.mean(X_current, axis=1)
                js_drift = self.jensen_shannon_divergence(ref_summary, curr_summary)
                data_source = "input_features"

            # Дрейф объяснений (уникально для Trust-ADE)
            explanation_drift = 0.0
            if explanations_reference is not None and explanations_current is not None:
                explanation_drift = self.explanation_quality_drift(
                    explanations_reference, explanations_current
                )

            # 🎯 Основная формула Trust-ADE
            concept_drift_rate = (self.lambda_param * ks_drift +
                                  (1 - self.lambda_param) * js_drift)

            # Интерпретация уровня дрейфа
            if concept_drift_rate < 0.1:
                drift_level = "Minimal"
            elif concept_drift_rate < 0.3:
                drift_level = "Low"
            elif concept_drift_rate < 0.5:
                drift_level = "Moderate"
            elif concept_drift_rate < 0.7:
                drift_level = "High"
            else:
                drift_level = "Critical"

            results = {
                'concept_drift_rate': np.clip(concept_drift_rate, 0.0, 1.0),
                'ks_drift': np.clip(ks_drift, 0.0, 1.0),
                'js_divergence': np.clip(js_drift, 0.0, 1.0),
                'explanation_drift': np.clip(explanation_drift, 0.0, 1.0),
                'drift_level': drift_level,
                'js_data_source': data_source,
                'lambda_param': self.lambda_param
            }

            if verbose:
                print(f"📊 Trust-ADE Concept Drift Results:")
                print(f"   🎯 Concept-Drift Rate: {results['concept_drift_rate']:.4f} ({drift_level})")
                print(f"   📈 KS Component (λ={self.lambda_param}): {results['ks_drift']:.4f}")
                print(f"   📊 JS Component (1-λ={1 - self.lambda_param}): {results['js_divergence']:.4f}")
                print(f"   🧠 Explanation Drift: {results['explanation_drift']:.4f}")
                print(f"   📋 JS Data Source: {data_source}")

            return results

        except Exception as e:
            warnings.warn(f"🚨 Критическая ошибка в ConceptDrift.calculate: {str(e)}")
            return self._default_results()

    def _default_results(self) -> Dict[str, float]:
        """Результаты по умолчанию при критических ошибках"""
        return {
            'concept_drift_rate': 0.5,
            'ks_drift': 0.5,
            'js_divergence': 0.5,
            'explanation_drift': 0.5,
            'drift_level': 'Unknown',
            'js_data_source': 'error',
            'lambda_param': self.lambda_param
        }