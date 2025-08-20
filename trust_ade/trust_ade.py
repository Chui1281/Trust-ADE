"""
Главный класс протокола Trust-ADE для комплексной оценки доверия к ИИ-системам
"""

import numpy as np
import warnings
import sys
import os

from .explainability_score import ExplainabilityScore
from .robustness_index import RobustnessIndex
from .bias_shift_index import BiasShiftIndex
from .concept_drift import ConceptDrift
from .trust_calculator import TrustCalculator
from .utils import validate_inputs, check_explainer_compatibility


class TrustADE:
    """
    Главный класс протокола Trust-ADE для оценки доверия к ИИ-системам

    Объединяет все компоненты оценки:
    - Explainability Score (объяснимость)
    - Robustness Index (устойчивость)
    - Bias Shift Index (справедливость)
    - Concept Drift Rate (стабильность во времени)
    """

    def __init__(self, model, domain='general', protected_attributes=None,
                 explainer_type='shap', expert_causal_graph=None, training_data=None):
        """
        Инициализация системы Trust-ADE

        Args:
            model: ML модель для оценки (должна реализовать BaseModel интерфейс)
            domain: домен применения ('medical', 'finance', 'criminal_justice', 'general')
            protected_attributes: список защищенных атрибутов для мониторинга справедливости
            explainer_type: тип объяснителя ('shap', 'lime', 'auto')
            expert_causal_graph: экспертный каузальный граф (опционально)
            training_data: обучающие данные для инициализации объяснителя
        """
        self.model = model
        self.domain = domain
        self.protected_attributes = protected_attributes or []
        self.expert_causal_graph = expert_causal_graph
        self.training_data = training_data

        # Инициализация компонентов оценки
        self.es_calculator = ExplainabilityScore()
        self.ri_calculator = RobustnessIndex()
        self.bsi_calculator = BiasShiftIndex(protected_attributes)
        self.cd_detector = ConceptDrift()
        self.trust_calc = TrustCalculator(domain)

        # Инициализация объяснителя
        self.explainer = self._create_explainer(explainer_type, training_data)

        # Кеш последних результатов
        self.last_results = None
        self._evaluation_history = []

    def _create_explainer(self, explainer_type, training_data):
        """Создание объяснителя с обработкой ошибок импорта"""
        try:
            # Динамический импорт для избежания циркулярных зависимостей
            if explainer_type == 'shap':
                try:
                    # Попытка импорта из локального модуля
                    from explainers.shap_explainer import SHAPExplainer
                    return SHAPExplainer(self.model, training_data)
                except ImportError:
                    # Fallback: создаем простой объяснитель
                    warnings.warn("Не удалось импортировать SHAP explainer, используется fallback")
                    return self._create_simple_explainer()
            else:
                warnings.warn(f"Неподдерживаемый тип explainer: {explainer_type}")
                return self._create_simple_explainer()

        except Exception as e:
            warnings.warn(f"Ошибка создания explainer: {str(e)}")
            return self._create_simple_explainer()

    def _create_simple_explainer(self):
        """Создание простого объяснителя как fallback"""

        class SimpleExplainer:
            def __init__(self, model):
                self.model = model

            def explain(self, X):
                try:
                    # Получаем важности признаков из модели
                    importance = self.model.get_feature_importance()
                    if importance is not None and len(importance) == X.shape[1]:
                        # Генерируем объяснения как произведение входов на важность
                        return X * importance.reshape(1, -1)
                    else:
                        # Fallback: равномерные веса
                        return X * (1.0 / X.shape[1])
                except Exception as e:
                    warnings.warn(f"Ошибка в simple explainer: {str(e)}")
                    return np.zeros_like(X)

            def shap_values(self, X):
                """Совместимость с SHAP интерфейсом"""
                return self.explain(X)

        return SimpleExplainer(self.model)

    def evaluate(self, X_test, y_test, X_reference=None, y_reference=None,
                 protected_data=None, expert_ratings=None, verbose=True):
        """
        Основная функция комплексной оценки доверия к ИИ-системе

        Args:
            X_test: тестовые входные данные
            y_test: тестовые целевые переменные
            X_reference: референсные данные для сравнения (опционально)
            y_reference: референсные целевые переменные (опционально)
            protected_data: данные о защищенных атрибутах (опционально)
            expert_ratings: экспертные оценки понятности (опционально)
            verbose: выводить ли промежуточные сообщения

        Returns:
            dict: полные результаты оценки доверия
        """
        try:
            if verbose:
                print("🔍 Запуск комплексной оценки Trust-ADE...")

            # Валидация входных данных
            X_test, y_test = validate_inputs(X_test, y_test)

            # Проверка совместимости explainer
            if not check_explainer_compatibility(self.explainer):
                warnings.warn("Explainer может быть несовместим")

            # 1. Вычисление Explainability Score
            if verbose:
                print("📊 Вычисляем Explainability Score...")

            es_results = self.es_calculator.calculate(
                self.model, self.explainer, X_test, y_test,
                expert_graph=self.expert_causal_graph,
                expert_ratings=expert_ratings
            )
            es = es_results['explainability_score']

            # 2. Вычисление Robustness Index
            if verbose:
                print("🛡️ Вычисляем Robustness Index...")

            ri_results = self.ri_calculator.calculate(
                self.model, self.explainer, X_test, y_test
            )
            ri = ri_results['robustness_index']

            # 3. Вычисление Bias Shift Index (только если есть базовые данные)
            bsi = 0.0
            bsi_results = {'bias_shift_index': 0.0}

            if X_reference is not None and protected_data is not None:
                if verbose:
                    print("⚖️ Вычисляем Bias Shift Index...")

                try:
                    y_pred_current = self.model.predict(X_test)
                    y_pred_baseline = self.model.predict(X_reference)

                    bsi_results = self.bsi_calculator.calculate(
                        y_test, y_pred_current, y_pred_baseline, protected_data
                    )
                    bsi = bsi_results['bias_shift_index']
                except Exception as e:
                    warnings.warn(f"Ошибка вычисления BSI: {str(e)}")

            # 4. Вычисление Concept Drift Rate (только если есть базовые данные)
            cdr = 0.0
            cdr_results = {'concept_drift_rate': 0.0}

            if X_reference is not None:
                if verbose:
                    print("📈 Вычисляем Concept Drift Rate...")

                try:
                    y_pred_current = self.model.predict(X_test)
                    y_pred_reference = self.model.predict(X_reference)

                    cdr_results = self.cd_detector.calculate(
                        X_reference, X_test, y_pred_reference, y_pred_current
                    )
                    cdr = cdr_results['concept_drift_rate']
                except Exception as e:
                    warnings.warn(f"Ошибка вычисления CDR: {str(e)}")

            # 5. Вычисление итогового Trust Score
            if verbose:
                print("🎯 Вычисляем итоговый Trust Score...")

            trust_results = self.trust_calc.calculate_trust_score(es, ri, bsi, cdr)

            # Объединение всех результатов
            final_results = {
                'trust_score': trust_results['trust_score'],
                'trust_level': self.trust_calc.get_trust_level_description(trust_results['trust_score']),
                'explainability_score': es,
                'robustness_index': ri,
                'bias_shift_index': bsi,
                'concept_drift_rate': cdr,
                'components': trust_results['components'],
                'weights_used': trust_results['weights_used'],
                'domain': self.domain,
                'detailed_results': {
                    'explainability_details': es_results,
                    'robustness_details': ri_results,
                    'bias_details': bsi_results,
                    'drift_details': cdr_results
                },
                'recommendations': self.trust_calc.get_recommendations(trust_results),
                'evaluation_timestamp': np.datetime64('now')
            }

            # Сохранение результатов
            self.last_results = final_results
            self._evaluation_history.append(final_results)

            if verbose:
                print(f"✅ Оценка завершена! Trust Score: {final_results['trust_score']:.3f}")
                print(f"📊 Уровень доверия: {final_results['trust_level']}")

            return final_results

        except Exception as e:
            error_msg = f"Критическая ошибка при выполнении оценки Trust-ADE: {str(e)}"
            warnings.warn(error_msg)

            # Возвращаем базовые результаты при ошибке
            error_results = {
                'trust_score': 0.0,
                'trust_level': 'Ошибка оценки',
                'explainability_score': 0.0,
                'robustness_index': 0.0,
                'bias_shift_index': 0.0,
                'concept_drift_rate': 0.0,
                'error': error_msg,
                'evaluation_timestamp': np.datetime64('now')
            }

            self.last_results = error_results
            return error_results

    def generate_report(self, results=None, output_file=None):
        """
        Генерация подробного отчета по результатам оценки

        Args:
            results: результаты оценки (если None, используются последние результаты)
            output_file: файл для сохранения отчета (опционально)

        Returns:
            str: текст отчета
        """
        if results is None:
            results = self.last_results

        if results is None:
            return "❌ Нет результатов для генерации отчета. Сначала выполните evaluate()."

        try:
            if 'error' in results:
                report = f"""
❌ ОТЧЕТ TRUST-ADE: ОШИБКА ВЫПОЛНЕНИЯ ❌

Ошибка: {results['error']}
Время: {results.get('evaluation_timestamp', 'Неизвестно')}

Рекомендуется проверить входные данные и настройки системы.
"""
            else:
                # Генерация полного отчета
                trust_score = results['trust_score']
                trust_level = results['trust_level']

                report = f"""
=== 📊 ОТЧЕТ TRUST-ADE ===

🎯 Общая оценка доверия: {trust_score:.3f}
📈 Уровень доверия: {trust_level}
🏷️ Домен применения: {results['domain']}
🕐 Время оценки: {results.get('evaluation_timestamp', 'Неизвестно')}

=== 📋 Детализация компонентов ===
• Explainability Score: {results['explainability_score']:.3f}
• Robustness Index: {results['robustness_index']:.3f}
• Bias Shift Index: {results['bias_shift_index']:.3f}
• Concept Drift Rate: {results['concept_drift_rate']:.3f}

=== ⚖️ Веса компонентов (домен: {results['domain']}) ===
• Объяснимость (w_E): {results['weights_used']['w_E']:.3f}
• Устойчивость (w_R): {results['weights_used']['w_R']:.3f}
• Справедливость (w_F): {results['weights_used']['w_F']:.3f}
• Чувствительность к дрейфу (γ): {results['weights_used']['gamma']:.1f}

=== 🔍 Детальный анализ объяснимости ===
• Каузальная фиделити: {results['detailed_results']['explainability_details']['causal_fidelity']:.3f}
• Семантическая когерентность: {results['detailed_results']['explainability_details']['semantic_coherence']:.3f}
• Стабильность интерпретаций: {results['detailed_results']['explainability_details']['interpretation_stability']:.3f}
• Человеческая понятность: {results['detailed_results']['explainability_details']['human_comprehensibility']:.3f}

=== 🛡️ Детальный анализ устойчивости ===
• Adversarial устойчивость: {results['detailed_results']['robustness_details']['adversarial_robustness']:.3f}
• Шумовая устойчивость: {results['detailed_results']['robustness_details']['noise_robustness']:.3f}
• Устойчивость объяснений: {results['detailed_results']['robustness_details']['explanation_robustness']:.3f}

=== 💡 Рекомендации ==="""

                for i, rec in enumerate(results['recommendations'], 1):
                    report += f"\n{i}. {rec}"

                report += "\n\n=== 📊 Вклад компонентов в итоговый счет ==="
                components = results['components']
                report += f"\n• От объяснимости: {components['explainability_component']:.3f}"
                report += f"\n• От устойчивости: {components['robustness_component']:.3f}"
                report += f"\n• От справедливости: {components['fairness_component']:.3f}"
                report += f"\n• Штраф за дрейф: {components['drift_penalty']:.3f}"

                report += "\n\n=========================="

            # Сохранение в файл при необходимости
            if output_file:
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"📄 Отчет сохранен в файл: {output_file}")
                except Exception as e:
                    warnings.warn(f"Ошибка сохранения отчета: {str(e)}")

            return report

        except Exception as e:
            error_report = f"❌ Ошибка генерации отчета: {str(e)}"
            warnings.warn(error_report)
            return error_report

    def get_trust_level(self, trust_score=None):
        """
        Получение текстового описания уровня доверия

        Args:
            trust_score: счет доверия (если None, используется последний результат)

        Returns:
            str: описание уровня доверия
        """
        if trust_score is None:
            if self.last_results is None:
                return "Оценка не проведена"
            trust_score = self.last_results.get('trust_score', 0.0)

        return self.trust_calc.get_trust_level_description(trust_score)

    def get_evaluation_history(self):
        """
        Получение истории оценок

        Returns:
            list: список всех проведенных оценок
        """
        return self._evaluation_history.copy()

    def calibrate_weights(self, expert_trust_ratings, evaluation_data):
        """
        Калибровка весов на основе экспертных оценок

        Args:
            expert_trust_ratings: список экспертных оценок доверия
            evaluation_data: список результатов оценки для калибровки

        Returns:
            dict: обновленные веса
        """
        try:
            # Извлекаем компоненты из результатов оценки
            computed_scores = []
            for data in evaluation_data:
                if isinstance(data, dict):
                    es = data.get('explainability_score', 0.5)
                    ri = data.get('robustness_index', 0.5)
                    bsi = data.get('bias_shift_index', 0.0)
                    cdr = data.get('concept_drift_rate', 0.0)
                else:
                    # Предполагаем, что data это кортеж (es, ri, bsi, cdr)
                    es, ri, bsi, cdr = data

                computed_scores.append((es, ri, bsi, cdr))

            return self.trust_calc.adaptive_weight_calibration(
                expert_trust_ratings, computed_scores
            )

        except Exception as e:
            warnings.warn(f"Ошибка калибровки весов: {str(e)}")
            return self.trust_calc.weights
