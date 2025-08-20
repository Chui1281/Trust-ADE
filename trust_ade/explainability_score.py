"""
Модуль для вычисления показателя объяснимости (Explainability Score)
"""

import numpy as np
import warnings
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from .utils import safe_explain, validate_inputs


class ExplainabilityScore:
    """
    Класс для вычисления показателя объяснимости
    Включает каузальную фиделити, семантическую когерентность, стабильность и понятность
    """

    def __init__(self, causal_weight=0.3, coherence_weight=0.3,
                 stability_weight=0.2, human_weight=0.2):
        """
        Инициализация с весами компонентов

        Args:
            causal_weight: вес каузальной фиделити
            coherence_weight: вес семантической когерентности
            stability_weight: вес стабильности интерпретаций
            human_weight: вес человеческой понятности
        """
        # Нормализуем веса
        total_weight = causal_weight + coherence_weight + stability_weight + human_weight

        self.causal_weight = causal_weight / total_weight
        self.coherence_weight = coherence_weight / total_weight
        self.stability_weight = stability_weight / total_weight
        self.human_weight = human_weight / total_weight

    def causal_fidelity(self, explanations, expert_graph=None):
        """
        Каузальная фиделити - соответствие объяснений реальным причинно-следственным связям

        Args:
            explanations: матрица объяснений
            expert_graph: экспертный каузальный граф (опционально)

        Returns:
            float: оценка каузальной фиделити от 0 до 1
        """
        try:
            if expert_graph is None:
                return self._heuristic_causal_fidelity(explanations)

            model_top = self._get_top_features(explanations, k=5)
            expert_top = set(expert_graph.get('important_features', []))

            if len(expert_top) == 0:
                return 0.5  # Нейтральная оценка

            intersection = len(model_top.intersection(expert_top))
            union = len(model_top.union(expert_top))

            if union == 0:
                return 0.5

            # Комбинируем precision и recall
            precision = intersection / len(model_top) if len(model_top) > 0 else 0
            recall = intersection / len(expert_top) if len(expert_top) > 0 else 0

            # F1-мера
            if precision + recall > 0:
                return 2 * precision * recall / (precision + recall)
            else:
                return 0

        except Exception as e:
            warnings.warn(f"Ошибка в causal_fidelity: {str(e)}")
            return 0.5

    def _heuristic_causal_fidelity(self, explanations):
        """Эвристическая оценка каузальности через консистентность"""
        try:
            if len(explanations) < 2:
                return 0.5

            # Ограничиваем количество объяснений для производительности
            sample_size = min(20, len(explanations))
            sample_explanations = explanations[:sample_size]

            # Вычисляем ранги важности признаков для каждого объяснения
            ranks = []
            for exp in sample_explanations:
                if len(exp) > 0:
                    rank = np.argsort(np.abs(exp))[::-1]
                    ranks.append(rank)

            if len(ranks) < 2:
                return 0.5

            # Вычисляем корреляции между рангами
            correlations = []
            for i in range(len(ranks)):
                for j in range(i + 1, len(ranks)):
                    try:
                        corr = np.corrcoef(ranks[i], ranks[j])[0, 1]
                        if not np.isnan(corr) and not np.isinf(corr):
                            correlations.append(abs(corr))
                    except:
                        continue

            return np.mean(correlations) if correlations else 0.5

        except Exception as e:
            warnings.warn(f"Ошибка в _heuristic_causal_fidelity: {str(e)}")
            return 0.5

    def _get_top_features(self, explanations, k=5):
        """Получение топ-k важных признаков"""
        try:
            mean_importance = np.mean(np.abs(explanations), axis=0)
            if len(mean_importance) < k:
                k = len(mean_importance)
            top_indices = np.argpartition(mean_importance, -k)[-k:]
            return set(top_indices)
        except Exception as e:
            warnings.warn(f"Ошибка в _get_top_features: {str(e)}")
            return set()

    def semantic_coherence(self, explanations):
        """
        Семантическая когерентность - логическая последовательность объяснений

        Args:
            explanations: матрица объяснений

        Returns:
            float: оценка семантической когерентности от 0 до 1
        """
        try:
            if len(explanations) == 0:
                return 0

            # Нормализуем объяснения
            exp_flat = np.array(explanations).flatten()
            exp_normalized = np.abs(exp_flat)

            total_mass = np.sum(exp_normalized)
            if total_mass < 1e-10:
                return 0.5  # Нейтральная оценка для нулевых объяснений

            exp_normalized = exp_normalized / total_mass
            exp_normalized = exp_normalized[exp_normalized > 1e-8]

            if len(exp_normalized) <= 1:
                return 1.0  # Максимальная когерентность для константных объяснений

            # Вычисляем энтропию
            H = entropy(exp_normalized)
            H_max = np.log(len(exp_normalized))

            # Инвертируем энтропию (меньше энтропии = больше когерентности)
            coherence = 1 - (H / H_max) if H_max > 0 else 1.0

            return max(0, min(1, coherence))

        except Exception as e:
            warnings.warn(f"Ошибка в semantic_coherence: {str(e)}")
            return 0.5

    def interpretation_stability(self, model, explainer, X, perturbation_size=0.01, n_samples=20):
        """
        Стабильность интерпретаций при небольших возмущениях входных данных

        Args:
            model: модель для предсказания
            explainer: объяснитель
            X: входные данные
            perturbation_size: размер возмущения
            n_samples: количество образцов для тестирования

        Returns:
            float: оценка стабильности от 0 до 1
        """
        try:
            if len(X) == 0:
                return 0

            stabilities = []
            n_test = min(n_samples, len(X))

            for i in range(n_test):
                try:
                    # Оригинальный образец
                    original_x = X[i:i + 1]

                    # Генерируем возмущенную версию
                    noise = np.random.normal(0, perturbation_size, original_x.shape)
                    perturbed_x = original_x + noise

                    # Получаем объяснения (используем safe_explain!)
                    original_exp = safe_explain(explainer, original_x)
                    perturbed_exp = safe_explain(explainer, perturbed_x)

                    # Проверяем, что объяснения не пустые
                    if len(original_exp) > 0 and len(perturbed_exp) > 0:
                        # Вычисляем косинусное сходство
                        similarity = 1 - cosine(original_exp[0], perturbed_exp)

                        # Ограничиваем диапазон [0, 1]
                        similarity = max(0, min(1, similarity))
                        stabilities.append(similarity)

                except Exception as e:
                    # Пропускаем проблемные образцы
                    continue

            return np.mean(stabilities) if stabilities else 0.5

        except Exception as e:
            warnings.warn(f"Ошибка в interpretation_stability: {str(e)}")
            return 0.5

    def human_comprehensibility(self, explanations, expert_ratings=None):
        """
        Человеческая понятность объяснений

        Args:
            explanations: матрица объяснений
            expert_ratings: экспертные оценки понятности (опционально)

        Returns:
            float: оценка понятности от 0 до 1
        """
        try:
            if expert_ratings is not None:
                return np.mean(expert_ratings)

            # Эвристическая оценка через разреженность
            # Более разреженные объяснения считаются более понятными
            sparsity_scores = []
            for exp in explanations:
                if len(exp) > 0:
                    non_zero_ratio = np.sum(np.abs(exp) > 1e-6) / len(exp)
                    # Инвертируем: меньше активных признаков = выше понятность
                    sparsity_score = 1 - min(1, non_zero_ratio)
                    sparsity_scores.append(sparsity_score)

            return np.mean(sparsity_scores) if sparsity_scores else 0.5

        except Exception as e:
            warnings.warn(f"Ошибка в human_comprehensibility: {str(e)}")
            return 0.5

    def calculate(self, model, explainer, X, y=None, expert_graph=None, expert_ratings=None):
        """
        Вычисление итогового Explainability Score

        Args:
            model: модель для предсказания
            explainer: объяснитель
            X: входные данные
            y: целевые переменные (опционально)
            expert_graph: экспертный каузальный граф (опционально)
            expert_ratings: экспертные оценки понятности (опционально)

        Returns:
            dict: словарь с результатами оценки
        """
        try:
            # Валидация входных данных
            X, y = validate_inputs(X, y)

            # Ограничиваем размер выборки для производительности
            n_samples = min(50, len(X))
            if n_samples < len(X):
                sample_indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X

            # Генерируем объяснения (используем safe_explain!)
            explanations = safe_explain(explainer, X_sample)

            if explanations is None or len(explanations) == 0:
                warnings.warn("Не удалось получить объяснения")
                return self._default_results()

            # Вычисляем компоненты
            fc = self.causal_fidelity(explanations, expert_graph)
            cs = self.semantic_coherence(explanations)
            si = self.interpretation_stability(model, explainer, X_sample)
            uh = self.human_comprehensibility(explanations, expert_ratings)

            # Итоговый Explainability Score
            es = (self.causal_weight * fc +
                  self.coherence_weight * cs +
                  self.stability_weight * si +
                  self.human_weight * uh)

            return {
                'explainability_score': max(0, min(1, es)),
                'causal_fidelity': max(0, min(1, fc)),
                'semantic_coherence': max(0, min(1, cs)),
                'interpretation_stability': max(0, min(1, si)),
                'human_comprehensibility': max(0, min(1, uh))
            }

        except Exception as e:
            warnings.warn(f"Ошибка в ExplainabilityScore.calculate: {str(e)}")
            return self._default_results()

    def _default_results(self):
        """Результаты по умолчанию при ошибках"""
        return {
            'explainability_score': 0.5,
            'causal_fidelity': 0.5,
            'semantic_coherence': 0.5,
            'interpretation_stability': 0.5,
            'human_comprehensibility': 0.5
        }
