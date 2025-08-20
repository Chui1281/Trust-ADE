"""
Модуль для вычисления итоговой метрики доверия
"""

import numpy as np
import warnings


class TrustCalculator:
    """
    Класс для вычисления итоговой метрики доверия Trust-ADE
    Объединяет объяснимость, устойчивость и справедливость в единый показатель
    """

    def __init__(self, domain='general'):
        """
        Инициализация калькулятора доверия

        Args:
            domain: домен применения ('medical', 'finance', 'criminal_justice', 'general')
        """
        self.domain = domain
        self.weights = self._load_domain_weights(domain)

    def _load_domain_weights(self, domain):
        """
        Загрузка весов для конкретного домена применения

        Args:
            domain: домен применения

        Returns:
            dict: словарь с весами и параметрами
        """
        domain_configs = {
            'medical': {
                'w_E': 0.5,  # Приоритет объяснимости в медицине
                'w_R': 0.3,  # Умеренная важность устойчивости
                'w_F': 0.2,  # Базовая важность справедливости
                'gamma': 2.0  # Высокая чувствительность к дрейфу
            },
            'finance': {
                'w_E': 0.3,  # Умеренная важность объяснимости
                'w_R': 0.4,  # Высокая важность устойчивости
                'w_F': 0.3,  # Важность справедливости
                'gamma': 1.5  # Средняя чувствительность к дрейфу
            },
            'criminal_justice': {
                'w_E': 0.3,  # Умеренная важность объяснимости
                'w_R': 0.2,  # Низкая важность устойчивости
                'w_F': 0.5,  # Максимальный приоритет справедливости
                'gamma': 2.5  # Очень высокая чувствительность к дрейфу
            },
            'industrial': {
                'w_E': 0.25,  # Низкая важность объяснимости
                'w_R': 0.5,  # Максимальный приоритет устойчивости
                'w_F': 0.25,  # Базовая важность справедливости
                'gamma': 1.0  # Низкая чувствительность к дрейфу
            },
            'general': {
                'w_E': 0.4,  # Сбалансированная важность объяснимости
                'w_R': 0.3,  # Сбалансированная важность устойчивости
                'w_F': 0.3,  # Сбалансированная важность справедливости
                'gamma': 1.0  # Средняя чувствительность к дрейфу
            }
        }

        config = domain_configs.get(domain, domain_configs['general'])

        # Нормализуем веса
        total_weight = config['w_E'] + config['w_R'] + config['w_F']
        config['w_E'] /= total_weight
        config['w_R'] /= total_weight
        config['w_F'] /= total_weight

        return config

    def calculate_trust_score(self, explainability_score, robustness_index,
                              bias_shift_index, concept_drift_rate):
        """
        Вычисление итогового Trust-ADE Score

        Formula: Trust_ADE = w_E × ES + w_R × (RI × e^(-γ × CDR)) + w_F × (1 - BSI)

        Args:
            explainability_score: оценка объяснимости [0, 1]
            robustness_index: индекс устойчивости [0, 1]
            bias_shift_index: индекс смещения предвзятости [0, 1]
            concept_drift_rate: скорость дрейфа концептов [0, 1]

        Returns:
            dict: результаты вычисления доверия
        """
        try:
            # Получаем веса и параметры
            w_E = self.weights['w_E']
            w_R = self.weights['w_R']
            w_F = self.weights['w_F']
            gamma = self.weights['gamma']

            # Ограничиваем входные значения диапазоном [0, 1]
            explainability_score = max(0, min(1, explainability_score))
            robustness_index = max(0, min(1, robustness_index))
            bias_shift_index = max(0, min(1, bias_shift_index))
            concept_drift_rate = max(0, min(1, concept_drift_rate))

            # Компоненты формулы Trust-ADE
            explainability_component = w_E * explainability_score

            # Устойчивость с экспоненциальным штрафом за дрейф
            drift_penalty = np.exp(-gamma * concept_drift_rate)
            robustness_component = w_R * (robustness_index * drift_penalty)

            # Справедливость (инвертируем bias_shift_index)
            fairness_component = w_F * (1 - bias_shift_index)

            # Итоговый счет доверия
            trust_score = explainability_component + robustness_component + fairness_component

            # Ограничиваем результат диапазоном [0, 1]
            trust_score = max(0, min(1, trust_score))

            return {
                'trust_score': trust_score,
                'components': {
                    'explainability_component': explainability_component,
                    'robustness_component': robustness_component,
                    'fairness_component': fairness_component,
                    'drift_penalty': drift_penalty
                },
                'weights_used': self.weights.copy(),
                'domain': self.domain
            }

        except Exception as e:
            warnings.warn(f"Ошибка в calculate_trust_score: {str(e)}")
            return {
                'trust_score': 0.5,
                'components': {
                    'explainability_component': 0.2,
                    'robustness_component': 0.15,
                    'fairness_component': 0.15,
                    'drift_penalty': 1.0
                },
                'weights_used': self.weights.copy(),
                'domain': self.domain
            }

    def adaptive_weight_calibration(self, expert_trust_ratings, computed_scores,
                                    learning_rate=0.01, regularization=0.1, max_iterations=100):
        """
        Адаптивная калибровка весов на основе экспертных оценок доверия

        Args:
            expert_trust_ratings: список экспертных оценок доверия [0, 1]
            computed_scores: список кортежей (ES, RI, BSI, CDR) для каждой оценки
            learning_rate: скорость обучения
            regularization: коэффициент регуляризации
            max_iterations: максимальное количество итераций

        Returns:
            dict: обновленные веса
        """
        try:
            if len(expert_trust_ratings) != len(computed_scores):
                raise ValueError("Количество экспертных оценок и вычисленных счетов должно совпадать")

            if len(expert_trust_ratings) < 2:
                warnings.warn("Недостаточно данных для калибровки весов")
                return self.weights

            # Начальные веса
            current_weights = np.array([self.weights['w_E'], self.weights['w_R'], self.weights['w_F']])
            gamma = self.weights['gamma']

            expert_ratings = np.array(expert_trust_ratings)

            best_weights = current_weights.copy()
            best_error = float('inf')

            for iteration in range(max_iterations):
                # Вычисляем предсказания с текущими весами
                predictions = []
                for es, ri, bsi, cdr in computed_scores:
                    # Ограничиваем значения
                    es = max(0, min(1, es))
                    ri = max(0, min(1, ri))
                    bsi = max(0, min(1, bsi))
                    cdr = max(0, min(1, cdr))

                    # Формула Trust-ADE
                    pred = (current_weights[0] * es +
                            current_weights[1] * (ri * np.exp(-gamma * cdr)) +
                            current_weights[2] * (1 - bsi))
                    predictions.append(max(0, min(1, pred)))

                predictions = np.array(predictions)

                # Вычисляем ошибку (MSE)
                error = np.mean((predictions - expert_ratings) ** 2)

                # Сохраняем лучшие веса
                if error < best_error:
                    best_error = error
                    best_weights = current_weights.copy()

                # Ранняя остановка при достижении приемлемой ошибки
                if error < 0.01:
                    break

                # Вычисляем градиент
                gradient = np.zeros(3)
                for i, (es, ri, bsi, cdr) in enumerate(computed_scores):
                    # Ограничиваем значения
                    es = max(0, min(1, es))
                    ri = max(0, min(1, ri))
                    bsi = max(0, min(1, bsi))
                    cdr = max(0, min(1, cdr))

                    pred_error = predictions[i] - expert_ratings[i]

                    # Частные производные по весам
                    gradient[0] += 2 * pred_error * es
                    gradient[1] += 2 * pred_error * (ri * np.exp(-gamma * cdr))
                    gradient[2] += 2 * pred_error * (1 - bsi)

                gradient /= len(computed_scores)

                # Добавляем регуляризацию
                gradient += regularization * current_weights

                # Обновляем веса
                current_weights -= learning_rate * gradient

                # Обеспечиваем неотрицательность весов
                current_weights = np.abs(current_weights)

                # Нормализуем веса
                weight_sum = np.sum(current_weights)
                if weight_sum > 0:
                    current_weights = current_weights / weight_sum
                else:
                    # Возвращаемся к исходным весам при проблемах
                    current_weights = np.array([self.weights['w_E'], self.weights['w_R'], self.weights['w_F']])
                    break

            # Обновляем веса лучшими найденными значениями
            self.weights['w_E'] = best_weights[0]
            self.weights['w_R'] = best_weights[1]
            self.weights['w_F'] = best_weights[2]

            return self.weights

        except Exception as e:
            warnings.warn(f"Ошибка в adaptive_weight_calibration: {str(e)}")
            return self.weights

    def get_trust_level_description(self, trust_score):
        """
        Получение текстового описания уровня доверия

        Args:
            trust_score: счет доверия [0, 1]

        Returns:
            str: описание уровня доверия
        """
        if trust_score >= 0.8:
            return "Высокое доверие"
        elif trust_score >= 0.6:
            return "Умеренное доверие"
        elif trust_score >= 0.4:
            return "Низкое доверие"
        elif trust_score >= 0.2:
            return "Очень низкое доверие"
        else:
            return "Критически низкое доверие"

    def get_recommendations(self, trust_results):
        """
        Получение рекомендаций по улучшению доверия

        Args:
            trust_results: результаты вычисления доверия

        Returns:
            list: список рекомендаций
        """
        recommendations = []

        try:
            components = trust_results.get('components', {})
            trust_score = trust_results.get('trust_score', 0.5)

            if trust_score < 0.6:
                recommendations.append("Общий уровень доверия требует улучшения")

            # Анализируем компоненты
            if components.get('explainability_component', 0) < 0.15:
                recommendations.append("Рекомендуется улучшить качество объяснений модели")

            if components.get('robustness_component', 0) < 0.1:
                recommendations.append("Необходимо повысить устойчивость модели к возмущениям")

            if components.get('fairness_component', 0) < 0.1:
                recommendations.append("Требуется работа над справедливостью модели")

            if components.get('drift_penalty', 1) < 0.8:
                recommendations.append("Обнаружен значительный дрейф концептов, рекомендуется переобучение")

            if not recommendations:
                recommendations.append("Модель демонстрирует хороший уровень доверия")

        except Exception as e:
            warnings.warn(f"Ошибка в get_recommendations: {str(e)}")
            recommendations.append("Не удалось сгенерировать рекомендации")

        return recommendations
