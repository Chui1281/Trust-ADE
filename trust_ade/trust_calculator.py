"""
Модуль для вычисления итоговой метрики доверия Trust-ADE
Версия 2.0 - полная интеграция с обновленной архитектурой
"""

import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional, Any


class TrustCalculator:
    """
    Класс для вычисления итоговой метрики доверия Trust-ADE
    Объединяет объяснимость, устойчивость и справедливость в единый показатель

    Версия 2.0 включает:
    - Verbose режим для детального логирования
    - Расширенную валидацию входных данных
    - Анализ трендов изменения доверия
    - Историю вычислений для мониторинга
    - Улучшенную адаптивную калибровку весов
    """

    def __init__(self, domain='general'):
        """
        Инициализация калькулятора доверия

        Args:
            domain: домен применения ('medical', 'finance', 'criminal_justice', 'industrial', 'general')
        """
        self.domain = domain
        self.weights = self._load_domain_weights(domain)
        self._trust_history = []  # История изменений доверия
        self._validation_enabled = True  # Включить валидацию по умолчанию

    def _load_domain_weights(self, domain: str) -> Dict[str, float]:
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
                'gamma': 2.0,  # Высокая чувствительность к дрейфу
                'description': 'Медицинские системы с приоритетом объяснимости'
            },
            'finance': {
                'w_E': 0.3,  # Умеренная важность объяснимости
                'w_R': 0.4,  # Высокая важность устойчивости
                'w_F': 0.3,  # Важность справедливости
                'gamma': 1.5,  # Средняя чувствительность к дрейфу
                'description': 'Финансовые системы с балансом всех метрик'
            },
            'criminal_justice': {
                'w_E': 0.3,  # Умеренная важность объяснимости
                'w_R': 0.2,  # Низкая важность устойчивости
                'w_F': 0.5,  # Максимальный приоритет справедливости
                'gamma': 2.5,  # Очень высокая чувствительность к дрейфу
                'description': 'Системы уголовной юстиции с приоритетом справедливости'
            },
            'industrial': {
                'w_E': 0.25,  # Низкая важность объяснимости
                'w_R': 0.5,  # Максимальный приоритет устойчивости
                'w_F': 0.25,  # Базовая важность справедливости
                'gamma': 1.0,  # Низкая чувствительность к дрейфу
                'description': 'Промышленные системы с приоритетом надежности'
            },
            'general': {
                'w_E': 0.4,  # Сбалансированная важность объяснимости
                'w_R': 0.3,  # Сбалансированная важность устойчивости
                'w_F': 0.3,  # Сбалансированная важность справедливости
                'gamma': 1.0,  # Средняя чувствительность к дрейфу
                'description': 'Универсальные системы с балансом метрик'
            }
        }

        config = domain_configs.get(domain, domain_configs['general'])

        # Нормализуем веса
        total_weight = config['w_E'] + config['w_R'] + config['w_F']
        if total_weight > 0:
            config['w_E'] /= total_weight
            config['w_R'] /= total_weight
            config['w_F'] /= total_weight
        else:
            warnings.warn("Сумма весов равна нулю, используются веса по умолчанию")
            config.update(domain_configs['general'])

        return config

    def _validate_inputs(self, explainability_score: float, robustness_index: float,
                        bias_shift_index: float, concept_drift_rate: float) -> bool:
        """
        Валидация входных параметров Trust-ADE

        Args:
            explainability_score: оценка объяснимости
            robustness_index: индекс устойчивости
            bias_shift_index: индекс смещения предвзятости
            concept_drift_rate: скорость дрейфа концептов

        Returns:
            bool: True если валидация пройдена

        Raises:
            ValueError: при критических ошибках валидации
        """
        if not self._validation_enabled:
            return True

        inputs = {
            'explainability_score': explainability_score,
            'robustness_index': robustness_index,
            'bias_shift_index': bias_shift_index,
            'concept_drift_rate': concept_drift_rate
        }

        for name, value in inputs.items():
            if value is None:
                raise ValueError(f"{name} не может быть None")

            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{name} должен быть числом, получен {type(value)}")

            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"{name} содержит недопустимые значения (NaN или Inf)")

            if not (0 <= value <= 1):
                warnings.warn(f"{name} = {value:.4f} выходит за диапазон [0,1], будет ограничен")

        return True

    def calculate_trust_score(self, explainability_score: float, robustness_index: float,
                            bias_shift_index: float, concept_drift_rate: float,
                            verbose: bool = False) -> Dict[str, Any]:
        """
        Вычисление итогового Trust-ADE Score согласно формуле из статьи:
        Trust_ADE = w_E × ES + w_R × (RI × e^(-γ × CDR)) + w_F × (1 - BSI)

        Args:
            explainability_score: оценка объяснимости [0, 1]
            robustness_index: индекс устойчивости [0, 1]
            bias_shift_index: индекс смещения предвзятости [0, 1]
            concept_drift_rate: скорость дрейфа концептов [0, 1]
            verbose: детальный вывод процесса вычисления

        Returns:
            dict: результаты вычисления доверия с расширенной информацией
        """
        try:
            if verbose:
                print(f"🎯 Trust-ADE расчет для домена '{self.domain}'")
                print(f"📋 Конфигурация: {self.weights.get('description', 'Стандартная')}")
                print(f"📊 Веса: ES={self.weights['w_E']:.3f}, RI={self.weights['w_R']:.3f}, F={self.weights['w_F']:.3f}")
                print(f"⚙️  Параметр γ (чувствительность к дрейфу): {self.weights['gamma']:.2f}")

            # Валидация входных данных
            self._validate_inputs(explainability_score, robustness_index,
                                bias_shift_index, concept_drift_rate)

            if verbose:
                print(f"📊 Входные метрики:")
                print(f"   ES (Объяснимость): {explainability_score:.3f}")
                print(f"   RI (Устойчивость): {robustness_index:.3f}")
                print(f"   BSI (Смещение): {bias_shift_index:.3f}")
                print(f"   CDR (Дрейф): {concept_drift_rate:.3f}")

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

            # Расчет дополнительных метрик
            trust_level = self.get_trust_level_description(trust_score)

            # Анализ качества компонентов
            component_quality = self._analyze_component_quality(
                explainability_component, robustness_component, fairness_component, drift_penalty
            )

            # Формирование результата
            result = {
                'trust_score': trust_score,
                'trust_level': trust_level,
                'trust_level_numeric': self._get_trust_level_numeric(trust_score),
                'components': {
                    'explainability_component': explainability_component,
                    'robustness_component': robustness_component,
                    'fairness_component': fairness_component,
                    'drift_penalty': drift_penalty
                },
                'component_quality': component_quality,
                'weights_used': self.weights.copy(),
                'domain': self.domain,
                'input_metrics': {
                    'explainability_score': explainability_score,
                    'robustness_index': robustness_index,
                    'bias_shift_index': bias_shift_index,
                    'concept_drift_rate': concept_drift_rate
                },
                'analysis_metadata': {
                    'validation_passed': True,
                    'weights_normalized': True,
                    'drift_impact': 1 - drift_penalty,
                    'dominant_component': max([
                        ('explainability', explainability_component),
                        ('robustness', robustness_component),
                        ('fairness', fairness_component)
                    ], key=lambda x: x[1])[0]
                },
                'timestamp': np.datetime64('now')
            }

            # Сохраняем в историю
            self._trust_history.append(result)

            # Ограничиваем размер истории (последние 100 записей)
            if len(self._trust_history) > 100:
                self._trust_history = self._trust_history[-50:]

            if verbose:
                print(f"🎯 Итоговый Trust Score: {trust_score:.3f}")
                print(f"📊 Компоненты:")
                print(f"   Объяснимость: {explainability_component:.3f} (вклад: {explainability_component/trust_score*100:.1f}%)")
                print(f"   Устойчивость: {robustness_component:.3f} (вклад: {robustness_component/trust_score*100:.1f}%)")
                print(f"   Справедливость: {fairness_component:.3f} (вклад: {fairness_component/trust_score*100:.1f}%)")
                print(f"🏆 Уровень доверия: {trust_level}")
                print(f"⚠️  Влияние дрейфа: {(1-drift_penalty)*100:.1f}% штрафа")
                print(f"🎯 Доминирующий компонент: {result['analysis_metadata']['dominant_component']}")

            return result

        except Exception as e:
            error_msg = f"Ошибка в calculate_trust_score: {str(e)}"
            warnings.warn(error_msg)

            # Возвращаем безопасный результат при ошибке
            return {
                'trust_score': 0.5,
                'trust_level': 'Неопределенное доверие (ошибка вычисления)',
                'trust_level_numeric': 2,
                'components': {
                    'explainability_component': 0.2,
                    'robustness_component': 0.15,
                    'fairness_component': 0.15,
                    'drift_penalty': 1.0
                },
                'component_quality': 'error',
                'weights_used': self.weights.copy(),
                'domain': self.domain,
                'error': error_msg,
                'timestamp': np.datetime64('now')
            }

    def _analyze_component_quality(self, expl_comp: float, robust_comp: float,
                                  fair_comp: float, drift_penalty: float) -> Dict[str, str]:
        """
        Анализ качества отдельных компонентов Trust-ADE

        Returns:
            dict: оценка качества каждого компонента
        """
        def quality_level(score: float, thresholds: List[float] = [0.8, 0.6, 0.4, 0.2]) -> str:
            if score >= thresholds[0]:
                return "отличное"
            elif score >= thresholds[1]:
                return "хорошее"
            elif score >= thresholds[2]:
                return "удовлетворительное"
            elif score >= thresholds[3]:
                return "низкое"
            else:
                return "критическое"

        return {
            'explainability_quality': quality_level(expl_comp),
            'robustness_quality': quality_level(robust_comp),
            'fairness_quality': quality_level(fair_comp),
            'drift_impact_quality': quality_level(drift_penalty),
            'overall_balance': 'сбалансированное' if max(expl_comp, robust_comp, fair_comp) - min(expl_comp, robust_comp, fair_comp) < 0.2 else 'несбалансированное'
        }

    def _get_trust_level_numeric(self, trust_score: float) -> int:
        """
        Получение числового уровня доверия для программного использования

        Returns:
            int: 4 - высокое, 3 - умеренное, 2 - низкое, 1 - очень низкое, 0 - критическое
        """
        if trust_score >= 0.8:
            return 4
        elif trust_score >= 0.6:
            return 3
        elif trust_score >= 0.4:
            return 2
        elif trust_score >= 0.2:
            return 1
        else:
            return 0

    def analyze_trust_trend(self, window_size: int = 5) -> Dict[str, Any]:
        """
        Анализ тренда изменения доверия за последние измерения

        Args:
            window_size: размер окна для анализа тренда

        Returns:
            dict: информация о тренде доверия
        """
        if len(self._trust_history) < 2:
            return {
                'trend': 'insufficient_data',
                'slope': 0.0,
                'message': 'Недостаточно данных для анализа тренда'
            }

        recent_scores = [entry['trust_score'] for entry in self._trust_history[-window_size:]]

        if len(recent_scores) >= 2:
            # Простая линейная регрессия для оценки тренда
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]

            # Определение тренда
            if slope > 0.02:
                trend = 'improving'
                trend_description = 'улучшается'
            elif slope < -0.02:
                trend = 'declining'
                trend_description = 'ухудшается'
            else:
                trend = 'stable'
                trend_description = 'стабильное'

            # Дополнительная статистика
            current_score = recent_scores[-1]
            previous_score = recent_scores[-2] if len(recent_scores) >= 2 else current_score
            change = current_score - previous_score

            return {
                'trend': trend,
                'trend_description': trend_description,
                'slope': slope,
                'recent_scores': recent_scores,
                'current_score': current_score,
                'previous_score': previous_score,
                'absolute_change': change,
                'relative_change': (change / previous_score * 100) if previous_score > 0 else 0,
                'window_size': len(recent_scores),
                'volatility': np.std(recent_scores) if len(recent_scores) > 1 else 0
            }

        return {'trend': 'stable', 'slope': 0.0}

    def get_trust_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Получение истории изменений доверия

        Args:
            limit: максимальное количество записей (None для всех)

        Returns:
            list: история вычислений Trust-ADE
        """
        if limit is None:
            return self._trust_history.copy()
        else:
            return self._trust_history[-limit:].copy() if self._trust_history else []

    def clear_history(self):
        """Очистка истории вычислений"""
        self._trust_history.clear()

    def adaptive_weight_calibration(self, expert_trust_ratings: List[float],
                                  computed_scores: List[Tuple[float, float, float, float]],
                                  learning_rate: float = 0.01, regularization: float = 0.1,
                                  max_iterations: int = 100, verbose: bool = False) -> Dict[str, float]:
        """
        Адаптивная калибровка весов на основе экспертных оценок доверия

        Args:
            expert_trust_ratings: список экспертных оценок доверия [0, 1]
            computed_scores: список кортежей (ES, RI, BSI, CDR) для каждой оценки
            learning_rate: скорость обучения
            regularization: коэффициент регуляризации
            max_iterations: максимальное количество итераций
            verbose: детальный вывод процесса калибровки

        Returns:
            dict: обновленные веса и статистика калибровки
        """
        try:
            if len(expert_trust_ratings) != len(computed_scores):
                raise ValueError("Количество экспертных оценок и вычисленных счетов должно совпадать")

            if len(expert_trust_ratings) < 2:
                warnings.warn("Недостаточно данных для калибровки весов")
                return self.weights

            if verbose:
                print(f"🔧 Начинаем адаптивную калибровку весов для домена '{self.domain}'")
                print(f"📊 Данные: {len(expert_trust_ratings)} экспертных оценок")

            # Начальные веса
            current_weights = np.array([self.weights['w_E'], self.weights['w_R'], self.weights['w_F']])
            gamma = self.weights['gamma']

            expert_ratings = np.array(expert_trust_ratings)
            best_weights = current_weights.copy()
            best_error = float('inf')
            error_history = []

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
                error_history.append(error)

                # Сохраняем лучшие веса
                if error < best_error:
                    best_error = error
                    best_weights = current_weights.copy()

                # Ранняя остановка при достижении приемлемой ошибки
                if error < 0.01:
                    if verbose:
                        print(f"✅ Достигнута целевая точность на итерации {iteration + 1}")
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
                    warnings.warn("Проблема с нормализацией весов, возвращение к исходным значениям")
                    break

                if verbose and (iteration + 1) % 20 == 0:
                    print(f"📊 Итерация {iteration + 1}: MSE = {error:.6f}, Веса = [{current_weights[0]:.3f}, {current_weights[1]:.3f}, {current_weights[2]:.3f}]")

            # Обновляем веса лучшими найденными значениями
            old_weights = self.weights.copy()
            self.weights['w_E'] = best_weights[0]
            self.weights['w_R'] = best_weights[1]
            self.weights['w_F'] = best_weights[2]

            if verbose:
                print(f"🎯 Калибровка завершена!")
                print(f"📈 Итоговая ошибка: {best_error:.6f}")
                print(f"🔄 Изменение весов:")
                print(f"   ES: {old_weights['w_E']:.3f} → {self.weights['w_E']:.3f}")
                print(f"   RI: {old_weights['w_R']:.3f} → {self.weights['w_R']:.3f}")
                print(f"   F:  {old_weights['w_F']:.3f} → {self.weights['w_F']:.3f}")

            return {
                **self.weights,
                'calibration_stats': {
                    'final_error': best_error,
                    'iterations_completed': len(error_history),
                    'error_history': error_history,
                    'improvement': (error_history[0] - best_error) / error_history * 100 if error_history else 0,
                    'converged': error < 0.01
                }
            }

        except Exception as e:
            warnings.warn(f"Ошибка в adaptive_weight_calibration: {str(e)}")
            return self.weights

    def get_trust_level_description(self, trust_score: float) -> str:
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

    def get_recommendations(self, trust_results: Dict[str, Any]) -> List[str]:
        """
        Получение рекомендаций по улучшению доверия

        Args:
            trust_results: результаты вычисления доверия

        Returns:
            list: список детальных рекомендаций
        """
        recommendations = []

        try:
            components = trust_results.get('components', {})
            trust_score = trust_results.get('trust_score', 0.5)
            domain = trust_results.get('domain', 'general')

            # Общий анализ
            if trust_score < 0.6:
                recommendations.append(f"🚨 Общий уровень доверия ({trust_score:.3f}) требует улучшения для домена '{domain}'")

            # Анализируем компоненты с учетом домена
            expl_comp = components.get('explainability_component', 0)
            robust_comp = components.get('robustness_component', 0)
            fair_comp = components.get('fairness_component', 0)
            drift_penalty = components.get('drift_penalty', 1)

            # Пороги для рекомендаций зависят от домена
            domain_thresholds = {
                'medical': {'expl': 0.25, 'robust': 0.15, 'fair': 0.10},
                'finance': {'expl': 0.15, 'robust': 0.20, 'fair': 0.15},
                'criminal_justice': {'expl': 0.15, 'robust': 0.10, 'fair': 0.25},
                'industrial': {'expl': 0.12, 'robust': 0.25, 'fair': 0.12},
                'general': {'expl': 0.15, 'robust': 0.15, 'fair': 0.15}
            }

            thresholds = domain_thresholds.get(domain, domain_thresholds['general'])

            # Рекомендации по объяснимости
            if expl_comp < thresholds['expl']:
                recommendations.append(
                    f"📊 Объяснимость ({expl_comp:.3f}) ниже порога для домена '{domain}'. "
                    "Рекомендации: улучшить каузальную точность объяснений, "
                    "добавить контекстную адаптацию, повысить семантическую когерентность"
                )

            # Рекомендации по устойчивости
            if robust_comp < thresholds['robust']:
                recommendations.append(
                    f"🛡️ Устойчивость ({robust_comp:.3f}) требует укрепления. "
                    "Рекомендации: провести adversarial training, улучшить noise tolerance, "
                    "стабилизировать объяснения при возмущениях данных"
                )

            # Рекомендации по справедливости
            if fair_comp < thresholds['fair']:
                recommendations.append(
                    f"⚖️ Справедливость ({fair_comp:.3f}) нуждается в улучшении. "
                    "Рекомендации: провести fairness audit, применить bias mitigation techniques, "
                    "обеспечить равенство объяснений для всех демографических групп"
                )

            # Рекомендации по дрейфу концептов
            if drift_penalty < 0.8:
                drift_impact = (1 - drift_penalty) * 100
                recommendations.append(
                    f"📈 Обнаружен значительный concept drift (штраф {drift_impact:.1f}%). "
                    "Рекомендации: переобучить модель на актуальных данных, "
                    "усилить мониторинг изменений, внедрить adaptive learning"
                )

            # Анализ баланса компонентов
            component_analysis = trust_results.get('component_quality', {})
            if component_analysis.get('overall_balance') == 'несбалансированное':
                recommendations.append(
                    "⚖️ Компоненты доверия несбалансированы. "
                    "Рекомендация: рассмотреть адаптивную калибровку весов "
                    "или корректировку доменных приоритетов"
                )

            # Специфичные для домена рекомендации
            domain_specific = self._get_domain_specific_recommendations(domain, trust_results)
            recommendations.extend(domain_specific)

            if not recommendations:
                recommendations.append("✅ Система демонстрирует хороший уровень доверия во всех аспектах")

        except Exception as e:
            warnings.warn(f"Ошибка в get_recommendations: {str(e)}")
            recommendations.append("⚠️ Не удалось сгенерировать персонализированные рекомендации")

        return recommendations

    def _get_domain_specific_recommendations(self, domain: str, trust_results: Dict[str, Any]) -> List[str]:
        """
        Специфичные для домена рекомендации

        Args:
            domain: домен приложения
            trust_results: результаты анализа доверия

        Returns:
            list: доменно-специфичные рекомендации
        """
        domain_recs = {
            'medical': [
                "🏥 Для медицинских систем: обеспечить соответствие FDA 21 CFR Part 820",
                "🔬 Провести клиническую валидацию объяснений с участием врачей",
                "📋 Документировать все решения для медицинского аудита"
            ],
            'finance': [
                "💰 Для финансовых систем: соответствие Basel III и MiFID II требованиям",
                "📊 Обеспечить real-time мониторинг рыночных рисков",
                "🔍 Провести stress-testing в различных рыночных условиях"
            ],
            'criminal_justice': [
                "⚖️ Для систем уголовной юстиции: критический аудит на предвзятость",
                "👥 Обеспечить equal treatment для всех демографических групп",
                "📜 Соблюдение конституционных принципов due process"
            ],
            'industrial': [
                "🏭 Для промышленных систем: соответствие ISO 26262 (функциональная безопасность)",
                "⚡ Оптимизировать для real-time производственных условий",
                "🔧 Интеграция с SCADA системами и промышленными протоколами"
            ]
        }

        return domain_recs.get(domain, [])

    def export_configuration(self) -> Dict[str, Any]:
        """
        Экспорт текущей конфигурации калькулятора

        Returns:
            dict: полная конфигурация для воспроизведения
        """
        return {
            'domain': self.domain,
            'weights': self.weights.copy(),
            'history_length': len(self._trust_history),
            'validation_enabled': self._validation_enabled,
            'version': '2.0',
            'export_timestamp': str(np.datetime64('now'))
        }

    def import_configuration(self, config: Dict[str, Any]):
        """
        Импорт конфигурации калькулятора

        Args:
            config: конфигурация для загрузки
        """
        try:
            self.domain = config.get('domain', 'general')
            if 'weights' in config:
                self.weights.update(config['weights'])
            self._validation_enabled = config.get('validation_enabled', True)
        except Exception as e:
            warnings.warn(f"Ошибка импорта конфигурации: {str(e)}")

    def set_validation_mode(self, enabled: bool):
        """
        Включение/отключение валидации входных данных

        Args:
            enabled: включить валидацию
        """
        self._validation_enabled = enabled

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Получение сводной статистики по истории вычислений

        Returns:
            dict: статистика использования калькулятора
        """
        if not self._trust_history:
            return {'message': 'Нет данных для анализа'}

        scores = [entry['trust_score'] for entry in self._trust_history]

        return {
            'total_evaluations': len(self._trust_history),
            'average_trust_score': np.mean(scores),
            'trust_score_std': np.std(scores),
            'min_trust_score': np.min(scores),
            'max_trust_score': np.max(scores),
            'domain': self.domain,
            'current_weights': self.weights.copy(),
            'trend_analysis': self.analyze_trust_trend(),
            'evaluation_period': {
                'first_evaluation': str(self._trust_history[0]['timestamp']),
                'last_evaluation': str(self._trust_history[-1]['timestamp'])
            } if self._trust_history else None
        }


# Пример использования для демонстрации всех возможностей
if __name__ == "__main__":
    # Создаем калькулятор для медицинского домена
    trust_calc = TrustCalculator(domain='medical')

    print("🎯 Демонстрация Trust-ADE Calculator v2.0\n")

    # Вычисляем Trust Score с подробным выводом
    result = trust_calc.calculate_trust_score(
        explainability_score=0.85,
        robustness_index=0.78,
        bias_shift_index=0.12,
        concept_drift_rate=0.05,
        verbose=True
    )

    print("\n📋 Рекомендации:")
    for rec in trust_calc.get_recommendations(result):
        print(f"  {rec}")

    print("\n📊 Экспорт конфигурации:")
    config = trust_calc.export_configuration()
    print(f"  Домен: {config['domain']}")
    print(f"  Веса: ES={config['weights']['w_E']:.3f}, RI={config['weights']['w_R']:.3f}, F={config['weights']['w_F']:.3f}")
