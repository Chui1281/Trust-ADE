"""
Модуль для вычисления индекса устойчивости (Robustness Index)
"""

import numpy as np
import warnings
from scipy.spatial.distance import cosine
from .utils import safe_explain, validate_inputs


class RobustnessIndex:
    """
    Класс для вычисления показателя устойчивости
    Включает устойчивость к adversarial атакам, шуму и устойчивость объяснений
    """

    def __init__(self, adversarial_weight=0.4, noise_weight=0.3, explanation_weight=0.3):
        """
        Инициализация с весами компонентов

        Args:
            adversarial_weight: вес устойчивости к adversarial атакам
            noise_weight: вес устойчивости к шуму
            explanation_weight: вес устойчивости объяснений
        """
        # Нормализуем веса
        total_weight = adversarial_weight + noise_weight + explanation_weight

        self.adversarial_weight = adversarial_weight / total_weight
        self.noise_weight = noise_weight / total_weight
        self.explanation_weight = explanation_weight / total_weight

    def adversarial_robustness(self, model, X, y, epsilon=0.1, n_samples=50):
        """
        Устойчивость к adversarial атакам (упрощенная версия)

        Args:
            model: модель для тестирования
            X: входные данные
            y: целевые переменные
            epsilon: размер возмущения
            n_samples: количество образцов для тестирования

        Returns:
            float: оценка устойчивости от 0 до 1
        """
        try:
            X, y = validate_inputs(X, y)

            n_test = min(n_samples, len(X))
            if n_test == 0:
                return 0.5

            sample_indices = np.random.choice(len(X), n_test, replace=False)
            X_sample = X[sample_indices]

            successful_attacks = 0
            valid_tests = 0

            for i in range(len(X_sample)):
                try:
                    # Оригинальное предсказание
                    original_pred = model.predict(X_sample[i:i + 1])[0]

                    # Генерируем adversarial пример (простой подход - случайный шум)
                    noise = np.random.uniform(-epsilon, epsilon, X_sample[i].shape)
                    adversarial_x = X_sample[i] + noise

                    # Предсказание для adversarial примера
                    adversarial_pred = model.predict(adversarial_x.reshape(1, -1))[0]

                    valid_tests += 1
                    if original_pred != adversarial_pred:
                        successful_attacks += 1

                except Exception as e:
                    # Пропускаем проблемные образцы
                    continue

            if valid_tests == 0:
                return 0.5

            # Робастность = 1 - доля успешных атак
            robustness = 1 - (successful_attacks / valid_tests)
            return max(0, min(1, robustness))

        except Exception as e:
            warnings.warn(f"Ошибка в adversarial_robustness: {str(e)}")
            return 0.5

    def noise_robustness(self, model, X, noise_levels=[0.01, 0.05, 0.1], n_samples=50):
        """
        Устойчивость к шуму в данных

        Args:
            model: модель для тестирования
            X: входные данные
            noise_levels: уровни шума для тестирования
            n_samples: количество образцов для тестирования

        Returns:
            float: оценка устойчивости от 0 до 1
        """
        try:
            X, _ = validate_inputs(X)

            n_test = min(n_samples, len(X))
            if n_test == 0:
                return 0.5

            sample_indices = np.random.choice(len(X), n_test, replace=False)
            X_sample = X[sample_indices]

            # Получаем оригинальные предсказания
            try:
                original_preds = model.predict(X_sample)
            except Exception as e:
                warnings.warn(f"Ошибка получения предсказаний: {str(e)}")
                return 0.5

            robustness_scores = []

            for noise_level in noise_levels:
                consistent_predictions = 0
                valid_tests = 0

                for i in range(len(X_sample)):
                    try:
                        # Добавляем шум
                        noise = np.random.normal(0, noise_level, X_sample[i].shape)
                        noisy_x = X_sample[i] + noise

                        # Предсказание для зашумленного образца
                        noisy_pred = model.predict(noisy_x.reshape(1, -1))[0]

                        valid_tests += 1
                        if original_preds[i] == noisy_pred:
                            consistent_predictions += 1

                    except Exception as e:
                        # Пропускаем проблемные образцы
                        continue

                if valid_tests > 0:
                    score = consistent_predictions / valid_tests
                    robustness_scores.append(score)

            return np.mean(robustness_scores) if robustness_scores else 0.5

        except Exception as e:
            warnings.warn(f"Ошибка в noise_robustness: {str(e)}")
            return 0.5

    def explanation_robustness(self, model, explainer, X, perturbation_size=0.01, n_samples=30):
        """
        Устойчивость объяснений к возмущениям

        Args:
            model: модель
            explainer: объяснитель
            X: входные данные
            perturbation_size: размер возмущения
            n_samples: количество образцов для тестирования

        Returns:
            float: оценка устойчивости объяснений от 0 до 1
        """
        try:
            X, _ = validate_inputs(X)

            n_test = min(n_samples, len(X))
            if n_test == 0:
                return 0.5

            sample_indices = np.random.choice(len(X), n_test, replace=False)
            X_sample = X[sample_indices]

            similarities = []

            for i in range(len(X_sample)):
                try:
                    # Оригинальные объяснения
                    original_exp = safe_explain(explainer, X_sample[i:i + 1])

                    if len(original_exp) == 0:
                        continue

                    # Генерируем возмущенные данные
                    noise = np.random.normal(0, perturbation_size, X_sample[i].shape)
                    perturbed_x = X_sample[i] + noise

                    # Объяснения для возмущенных данных
                    perturbed_exp = safe_explain(explainer, perturbed_x.reshape(1, -1))

                    if len(perturbed_exp) == 0:
                        continue

                    # Вычисляем косинусное сходство
                    similarity = 1 - cosine(original_exp[0], perturbed_exp)

                    # Ограничиваем диапазон [0, 1]
                    similarity = max(0, min(1, similarity))
                    similarities.append(similarity)

                except Exception as e:
                    # Пропускаем проблемные образцы
                    continue

            return np.mean(similarities) if similarities else 0.5

        except Exception as e:
            warnings.warn(f"Ошибка в explanation_robustness: {str(e)}")
            return 0.5

    def calculate(self, model, explainer, X, y):
        """
        Вычисление итогового Robustness Index

        Args:
            model: модель для тестирования
            explainer: объяснитель
            X: входные данные
            y: целевые переменные

        Returns:
            dict: словарь с результатами оценки
        """
        try:
            # Валидация входных данных
            X, y = validate_inputs(X, y)

            # Вычисляем компоненты
            r_a = self.adversarial_robustness(model, X, y)
            r_n = self.noise_robustness(model, X)
            r_e = self.explanation_robustness(model, explainer, X)

            # Итоговый индекс устойчивости
            ri = (self.adversarial_weight * r_a +
                  self.noise_weight * r_n +
                  self.explanation_weight * r_e)

            return {
                'robustness_index': max(0, min(1, ri)),
                'adversarial_robustness': max(0, min(1, r_a)),
                'noise_robustness': max(0, min(1, r_n)),
                'explanation_robustness': max(0, min(1, r_e))
            }

        except Exception as e:
            warnings.warn(f"Ошибка в RobustnessIndex.calculate: {str(e)}")
            return {
                'robustness_index': 0.5,
                'adversarial_robustness': 0.5,
                'noise_robustness': 0.5,
                'explanation_robustness': 0.5
            }
