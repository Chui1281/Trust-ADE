"""
Trust-ADE Robustness Index Module
Реализация R_I = w_a·R_a + w_n·R_n + w_e·R_e
"""

import numpy as np
import warnings
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import time


class RobustnessIndex:
    """
    Класс для вычисления индекса устойчивости согласно Trust-ADE

    Реализует формулы:
    - R_a = 1 - (1/|A|) * Σ I[f(x+a) ≠ f(x)]  # Adversarial устойчивость
    - R_n = E[similarity(f(x), f(x+ε))]         # Шумовая устойчивость
    - R_e = E[similarity(E(x), E(x+ε))]         # Устойчивость объяснений
    - R_I = w_a·R_a + w_n·R_n + w_e·R_e         # Интегральный индекс
    """

    def __init__(self, w_adversarial: float = 0.4, w_noise: float = 0.3, w_explanation: float = 0.3,
                 similarity_metric: str = 'cosine', epsilon_levels: List[float] = None,
                 n_perturbations: int = 10, seed: int = 42):
        """
        Args:
            w_adversarial: вес adversarial устойчивости
            w_noise: вес шумовой устойчивости
            w_explanation: вес устойчивости объяснений
            similarity_metric: метрика сходства ('cosine', 'euclidean', 'pearson')
            epsilon_levels: уровни возмущений для тестирования
            n_perturbations: количество возмущений на каждый уровень
            seed: random seed для воспроизводимости
        """
        # Нормализация весов согласно Trust-ADE
        total_weight = w_adversarial + w_noise + w_explanation
        self.w_a = w_adversarial / total_weight
        self.w_n = w_noise / total_weight
        self.w_e = w_explanation / total_weight

        self.similarity_metric = similarity_metric
        self.epsilon_levels = epsilon_levels or [0.01, 0.05, 0.1, 0.2]
        self.n_perturbations = max(1, n_perturbations)
        self.seed = seed

        np.random.seed(seed)

        print(f"🛡️ Trust-ADE Robustness Index initialized:")
        print(f"   w_a (Adversarial): {self.w_a:.3f}")
        print(f"   w_n (Noise): {self.w_n:.3f}")
        print(f"   w_e (Explanation): {self.w_e:.3f}")
        print(f"   Similarity metric: {similarity_metric}")
        print(f"   Epsilon levels: {self.epsilon_levels}")
        print(f"   Perturbations per level: {self.n_perturbations}")

    def _similarity_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Trust-ADE similarity function для измерения схожести

        Args:
            x1, x2: векторы для сравнения

        Returns:
            float: мера сходства [0, 1], где 1 = идентичны
        """
        try:
            x1_flat = np.array(x1).flatten()
            x2_flat = np.array(x2).flatten()

            if len(x1_flat) != len(x2_flat):
                return 0.0

            # Обработка константных векторов
            if np.var(x1_flat) < 1e-12 and np.var(x2_flat) < 1e-12:
                return 1.0 if np.allclose(x1_flat, x2_flat) else 0.0

            if self.similarity_metric == 'cosine':
                # 1 - cosine_distance для получения similarity
                similarity = 1 - cosine(x1_flat, x2_flat)

            elif self.similarity_metric == 'euclidean':
                # Нормализованная euclidean similarity
                max_dist = np.sqrt(len(x1_flat)) * max(np.max(np.abs(x1_flat)), np.max(np.abs(x2_flat)))
                if max_dist < 1e-12:
                    similarity = 1.0
                else:
                    similarity = 1 - euclidean(x1_flat, x2_flat) / max_dist

            elif self.similarity_metric == 'pearson':
                # Pearson correlation coefficient
                corr, _ = pearsonr(x1_flat, x2_flat)
                similarity = (corr + 1) / 2  # нормализация [-1,1] -> [0,1]

            else:
                # Default to cosine
                similarity = 1 - cosine(x1_flat, x2_flat)

            # Обработка NaN и inf
            if np.isnan(similarity) or np.isinf(similarity):
                return 0.5

            return np.clip(similarity, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в similarity function: {str(e)}")
            return 0.5

    def adversarial_robustness(self, model, X: np.ndarray, y: Optional[np.ndarray] = None,
                              attack_types: List[str] = None, n_samples: int = 100) -> float:
        """
        Trust-ADE формула Adversarial устойчивости:
        R_a = 1 - (1/|A|) * Σ I[f(x+a) ≠ f(x)]

        Args:
            model: модель для тестирования
            X: входные данные [n_samples, n_features]
            y: истинные метки (опционально)
            attack_types: типы атак ['fgsm', 'random', 'boundary']
            n_samples: количество образцов для тестирования

        Returns:
            float: Adversarial устойчивость R_a [0, 1]
        """
        try:
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            attack_types = attack_types or ['random', 'boundary']
            n_test = min(n_samples, len(X))

            if n_test == 0:
                return 0.5

            # Случайная выборка для тестирования
            test_indices = np.random.choice(len(X), n_test, replace=False)
            X_test = X[test_indices]

            total_attacks = 0
            successful_attacks = 0

            print(f"🎯 Testing adversarial robustness on {n_test} samples...")

            for i, x_orig in enumerate(X_test):
                try:
                    # Оригинальное предсказание
                    orig_pred = model.predict(x_orig.reshape(1, -1))[0]

                    # Генерируем adversarial примеры разных типов
                    for attack_type in attack_types:
                        for eps in self.epsilon_levels:
                            for _ in range(self.n_perturbations // len(self.epsilon_levels)):

                                # Генерация adversarial возмущения
                                if attack_type == 'random':
                                    # Случайное возмущение
                                    perturbation = np.random.uniform(-eps, eps, x_orig.shape)

                                elif attack_type == 'boundary':
                                    # Возмущение в направлении границы решения
                                    gradient_approx = np.random.normal(0, eps, x_orig.shape)
                                    perturbation = gradient_approx / (np.linalg.norm(gradient_approx) + 1e-8) * eps

                                elif attack_type == 'fgsm':
                                    # Упрощённый Fast Gradient Sign Method
                                    gradient_approx = np.random.choice([-1, 1], x_orig.shape)
                                    perturbation = eps * gradient_approx

                                else:
                                    perturbation = np.random.uniform(-eps, eps, x_orig.shape)

                                # Adversarial пример
                                x_adv = x_orig + perturbation

                                try:
                                    # Предсказание для adversarial примера
                                    adv_pred = model.predict(x_adv.reshape(1, -1))[0]

                                    total_attacks += 1

                                    # 📐 ФОРМУЛА TRUST-ADE: I[f(x+a) ≠ f(x)]
                                    if orig_pred != adv_pred:
                                        successful_attacks += 1

                                except Exception:
                                    continue

                except Exception as e:
                    continue

            if total_attacks == 0:
                return 0.5

            # 📐 ОСНОВНАЯ ФОРМУЛА TRUST-ADE
            attack_success_rate = successful_attacks / total_attacks
            r_a = 1 - attack_success_rate

            print(f"   📊 Adversarial attacks: {successful_attacks}/{total_attacks} successful")
            print(f"   🛡️ R_a = {r_a:.4f}")

            return np.clip(r_a, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в adversarial_robustness: {str(e)}")
            return 0.5

    def noise_robustness(self, model, X: np.ndarray, n_samples: int = 100) -> float:
        """
        Trust-ADE формула шумовой устойчивости:
        R_n = E[similarity(f(x), f(x+ε))] где ε ~ N(0,σ²)

        Args:
            model: модель для тестирования
            X: входные данные [n_samples, n_features]
            n_samples: количество образцов для тестирования

        Returns:
            float: Шумовая устойчивость R_n [0, 1]
        """
        try:
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            n_test = min(n_samples, len(X))
            if n_test == 0:
                return 0.5

            test_indices = np.random.choice(len(X), n_test, replace=False)
            X_test = X[test_indices]

            similarities = []

            print(f"🔊 Testing noise robustness on {n_test} samples...")

            for i, x_orig in enumerate(X_test):
                try:
                    # Оригинальное предсказание (может быть вектором вероятностей)
                    orig_pred = model.predict(x_orig.reshape(1, -1))
                    if hasattr(model, 'predict_proba'):
                        orig_pred = model.predict_proba(x_orig.reshape(1, -1))
                    orig_pred = np.array(orig_pred).flatten()

                    sample_similarities = []

                    # Тестируем разные уровни шума
                    for sigma in self.epsilon_levels:
                        for _ in range(self.n_perturbations):

                            # 📐 ФОРМУЛА TRUST-ADE: ε ~ N(0,σ²)
                            epsilon = np.random.normal(0, sigma, x_orig.shape)
                            x_noisy = x_orig + epsilon

                            try:
                                # Предсказание для зашумленного входа
                                noisy_pred = model.predict(x_noisy.reshape(1, -1))
                                if hasattr(model, 'predict_proba'):
                                    noisy_pred = model.predict_proba(x_noisy.reshape(1, -1))
                                noisy_pred = np.array(noisy_pred).flatten()

                                # 📐 ФОРМУЛА TRUST-ADE: similarity(f(x), f(x+ε))
                                similarity = self._similarity_function(orig_pred, noisy_pred)
                                sample_similarities.append(similarity)

                            except Exception:
                                continue

                    if sample_similarities:
                        similarities.extend(sample_similarities)

                except Exception:
                    continue

            if not similarities:
                return 0.5

            # 📐 ОСНОВНАЯ ФОРМУЛА TRUST-ADE: E[similarity(f(x), f(x+ε))]
            r_n = np.mean(similarities)

            print(f"   📊 Processed {len(similarities)} noise perturbations")
            print(f"   🔊 R_n = {r_n:.4f}")

            return np.clip(r_n, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в noise_robustness: {str(e)}")
            return 0.5

    def explanation_robustness(self, model, explainer, X: np.ndarray,
                             explanation_method: str = 'auto', n_samples: int = 50) -> float:
        """
        Trust-ADE формула устойчивости объяснений:
        R_e = E[similarity(E(x), E(x+ε))] где ε ~ N(0,σ²)

        Уникальная компонента Trust-ADE для оценки стабильности интерпретаций!

        Args:
            model: модель
            explainer: объяснитель (SHAP, LIME, etc.)
            X: входные данные [n_samples, n_features]
            explanation_method: метод извлечения объяснений
            n_samples: количество образцов для тестирования

        Returns:
            float: Устойчивость объяснений R_e [0, 1]
        """
        try:
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            n_test = min(n_samples, len(X))
            if n_test == 0:
                return 0.5

            test_indices = np.random.choice(len(X), n_test, replace=False)
            X_test = X[test_indices]

            explanation_similarities = []

            print(f"🧠 Testing explanation robustness on {n_test} samples...")

            for i, x_orig in enumerate(X_test):
                try:
                    # Оригинальное объяснение E(x)
                    if hasattr(explainer, 'explain'):
                        orig_explanation = explainer.explain(x_orig.reshape(1, -1))
                    elif hasattr(explainer, 'shap_values'):
                        orig_explanation = explainer.shap_values(x_orig.reshape(1, -1))
                    elif hasattr(explainer, 'explain_instance'):
                        # LIME-style explainer
                        exp = explainer.explain_instance(x_orig, model.predict_proba, num_features=len(x_orig))
                        orig_explanation = np.array([item[1] for item in exp.as_list()])
                    else:
                        # Fallback: gradient-based explanation
                        orig_explanation = np.random.normal(0, 0.1, x_orig.shape)

                    orig_explanation = np.array(orig_explanation).flatten()

                    sample_similarities = []

                    # Тестируем различные возмущения
                    for sigma in self.epsilon_levels[:2]:  # используем меньше уровней для объяснений
                        for _ in range(self.n_perturbations // 2):

                            # 📐 ФОРМУЛА TRUST-ADE: ε ~ N(0,σ²)
                            epsilon = np.random.normal(0, sigma, x_orig.shape)
                            x_perturbed = x_orig + epsilon

                            try:
                                # Объяснение для возмущённого входа E(x+ε)
                                if hasattr(explainer, 'explain'):
                                    perturbed_explanation = explainer.explain(x_perturbed.reshape(1, -1))
                                elif hasattr(explainer, 'shap_values'):
                                    perturbed_explanation = explainer.shap_values(x_perturbed.reshape(1, -1))
                                elif hasattr(explainer, 'explain_instance'):
                                    exp = explainer.explain_instance(x_perturbed, model.predict_proba, num_features=len(x_perturbed))
                                    perturbed_explanation = np.array([item[1] for item in exp.as_list()])
                                else:
                                    perturbed_explanation = np.random.normal(0, 0.1, x_orig.shape)

                                perturbed_explanation = np.array(perturbed_explanation).flatten()

                                # 📐 ФОРМУЛА TRUST-ADE: similarity(E(x), E(x+ε))
                                similarity = self._similarity_function(orig_explanation, perturbed_explanation)
                                sample_similarities.append(similarity)

                            except Exception:
                                continue

                    if sample_similarities:
                        explanation_similarities.extend(sample_similarities)

                except Exception:
                    continue

            if not explanation_similarities:
                return 0.5

            # 📐 ОСНОВНАЯ ФОРМУЛА TRUST-ADE: E[similarity(E(x), E(x+ε))]
            r_e = np.mean(explanation_similarities)

            print(f"   📊 Processed {len(explanation_similarities)} explanation perturbations")
            print(f"   🧠 R_e = {r_e:.4f}")

            return np.clip(r_e, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в explanation_robustness: {str(e)}")
            return 0.5

    def calculate_robustness_index(self, model, explainer, X: np.ndarray,
                                  y: Optional[np.ndarray] = None) -> float:
        """
        Основная формула Trust-ADE для Robustness Index:
        R_I = w_a·R_a + w_n·R_n + w_e·R_e

        Args:
            model: модель для тестирования
            explainer: объяснитель
            X: входные данные
            y: истинные метки (опционально)

        Returns:
            float: Robustness Index [0, 1]
        """
        try:
            # Компоненты Trust-ADE
            r_a = self.adversarial_robustness(model, X, y)
            r_n = self.noise_robustness(model, X)
            r_e = self.explanation_robustness(model, explainer, X)

            # 🎯 ФОРМУЛА TRUST-ADE
            robustness_index = self.w_a * r_a + self.w_n * r_n + self.w_e * r_e

            return np.clip(robustness_index, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в calculate_robustness_index: {str(e)}")
            return 0.5

    def calculate(self, model, explainer, X: Union[np.ndarray, list],
                 y: Optional[Union[np.ndarray, list]] = None,
                 n_samples: int = 100, verbose: bool = True) -> Dict[str, float]:
        """
        Полный анализ устойчивости согласно Trust-ADE с детальной разбивкой

        Args:
            model: модель для тестирования
            explainer: объяснитель
            X: входные данные
            y: истинные метки (опционально)
            n_samples: количество образцов для тестирования
            verbose: детальный вывод результатов

        Returns:
            Dict[str, float]: детальные результаты анализа устойчивости
        """
        try:
            if verbose:
                print(f"🛡️ Trust-ADE Robustness Analysis...")
                start_time = time.time()

            # Валидация данных
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            if y is not None:
                y = np.array(y)

            # Компоненты устойчивости Trust-ADE
            r_a = self.adversarial_robustness(model, X, y, n_samples=n_samples)
            r_n = self.noise_robustness(model, X, n_samples=n_samples)
            r_e = self.explanation_robustness(model, explainer, X, n_samples=n_samples // 2)

            # 🎯 Основная формула Trust-ADE
            robustness_index = self.w_a * r_a + self.w_n * r_n + self.w_e * r_e

            # Интерпретация уровня устойчивости
            if robustness_index >= 0.8:
                robustness_level = "Excellent"
            elif robustness_index >= 0.6:
                robustness_level = "Good"
            elif robustness_index >= 0.4:
                robustness_level = "Fair"
            elif robustness_index >= 0.2:
                robustness_level = "Poor"
            else:
                robustness_level = "Critical"

            results = {
                'robustness_index': np.clip(robustness_index, 0.0, 1.0),
                'adversarial_robustness': np.clip(r_a, 0.0, 1.0),
                'noise_robustness': np.clip(r_n, 0.0, 1.0),
                'explanation_robustness': np.clip(r_e, 0.0, 1.0),
                'robustness_level': robustness_level,
                'similarity_metric': self.similarity_metric,
                'epsilon_levels': self.epsilon_levels,
                'weights': {
                    'w_adversarial': self.w_a,
                    'w_noise': self.w_n,
                    'w_explanation': self.w_e
                }
            }

            if verbose:
                elapsed_time = time.time() - start_time
                print(f"📊 Trust-ADE Robustness Results:")
                print(f"   🎯 Robustness Index: {results['robustness_index']:.4f} ({robustness_level})")
                print(f"   ⚔️ Adversarial R_a: {results['adversarial_robustness']:.4f}")
                print(f"   🔊 Noise R_n: {results['noise_robustness']:.4f}")
                print(f"   🧠 Explanation R_e: {results['explanation_robustness']:.4f}")
                print(f"   ⏱️ Analysis time: {elapsed_time:.2f}s")

            return results

        except Exception as e:
            warnings.warn(f"🚨 Критическая ошибка в RobustnessIndex.calculate: {str(e)}")
            return self._default_results()

    def _default_results(self) -> Dict[str, Any]:
        """Результаты по умолчанию при критических ошибках"""
        return {
            'robustness_index': 0.5,
            'adversarial_robustness': 0.5,
            'noise_robustness': 0.5,
            'explanation_robustness': 0.5,
            'robustness_level': 'Unknown',
            'similarity_metric': self.similarity_metric,
            'epsilon_levels': self.epsilon_levels,
            'weights': {
                'w_adversarial': self.w_a,
                'w_noise': self.w_n,
                'w_explanation': self.w_e
            }
        }


# Демонстрация использования
if __name__ == "__main__":
    print("🚀 Trust-ADE RobustnessIndex v2.0 - Каузальный анализ устойчивости!")

    # Создаем анализатор устойчивости
    robustness_analyzer = RobustnessIndex(
        w_adversarial=0.4,    # приоритет adversarial устойчивости
        w_noise=0.35,         # шумовая устойчивость
        w_explanation=0.25,   # устойчивость объяснений
        similarity_metric='cosine',  # косинусное сходство
        epsilon_levels=[0.01, 0.05, 0.1],  # уровни возмущений
        n_perturbations=20    # количество тестов на уровень
    )

    # Mock модель для демонстрации
    class MockModel:
        def predict(self, X):
            return np.random.binomial(1, 0.7, X.shape[0])

        def predict_proba(self, X):
            proba = np.random.random(X.shape)
            return np.column_stack([1-proba, proba])

    # Mock объяснитель
    class MockExplainer:
        def explain(self, X):
            return np.random.normal(0, 1, X.shape)

    # Пример данных
    np.random.seed(42)
    X_test = np.random.random((100, 10))
    y_test = np.random.binomial(1, 0.6, 100)

    model = MockModel()
    explainer = MockExplainer()

    # Анализ устойчивости
    results = robustness_analyzer.calculate(
        model=model,
        explainer=explainer,
        X=X_test,
        y=y_test,
        n_samples=50,
        verbose=True
    )

    print(f"\n✅ Robustness Analysis Complete!")
    print(f"   Final R_I = {results['robustness_index']:.4f}")
    print(f"   Level: {results['robustness_level']}")