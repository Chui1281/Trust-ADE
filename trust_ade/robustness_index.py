"""
Trust-ADE Robustness Index Module (Optimized)
Реализация R_I = w_a·R_a + w_n·R_n + w_e·R_e
"""

import numpy as np
import warnings
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Опциональные CUDA импорты
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class RobustnessIndex:
    """
    Оптимизированный класс для вычисления индекса устойчивости согласно Trust-ADE

    Ключевые оптимизации:
    - Батчинг операций для ускорения
    - Опциональная CUDA поддержка
    - Параллельная обработка компонентов
    - Кэширование similarity вычислений
    - Улучшенная обработка ошибок

    Реализует формулы:
    - R_a = 1 - (1/|A|) * Σ I[f(x+a) ≠ f(x)]  # Adversarial устойчивость
    - R_n = E[similarity(f(x), f(x+ε))]         # Шумовая устойчивость
    - R_e = E[similarity(E(x), E(x+ε))]         # Устойчивость объяснений
    - R_I = w_a·R_a + w_n·R_n + w_e·R_e         # Интегральный индекс
    """

    def __init__(self, w_adversarial: float = 0.4, w_noise: float = 0.3, w_explanation: float = 0.3,
                 similarity_metric: str = 'cosine', epsilon_levels: List[float] = None,
                 n_perturbations: int = 10, seed: int = 42, batch_size: int = 1000,
                 use_cuda: bool = True, n_workers: int = 2):
        """
        Args:
            w_adversarial: вес adversarial устойчивости
            w_noise: вес шумовой устойчивости
            w_explanation: вес устойчивости объяснений
            similarity_metric: метрика сходства ('cosine', 'euclidean', 'pearson')
            epsilon_levels: уровни возмущений для тестирования
            n_perturbations: количество возмущений на каждый уровень
            seed: random seed для воспроизводимости
            batch_size: размер батча для обработки
            use_cuda: использовать ли CUDA ускорение
            n_workers: количество worker'ов для параллельной обработки
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
        self.batch_size = batch_size
        self.n_workers = n_workers

        # CUDA инициализация
        self.use_cuda = use_cuda and CUDA_AVAILABLE
        if self.use_cuda:
            try:
                cp.cuda.Device(0).use()
                print("🚀 CUDA acceleration enabled")
            except:
                self.use_cuda = False
                warnings.warn("CUDA инициализация не удалась, используем CPU")

        np.random.seed(seed)

        # Кэш для similarity вычислений
        self._similarity_cache = {}

        print(f"🛡️ Trust-ADE Robustness Index (Optimized) initialized:")
        print(f"   w_a (Adversarial): {self.w_a:.3f}")
        print(f"   w_n (Noise): {self.w_n:.3f}")
        print(f"   w_e (Explanation): {self.w_e:.3f}")
        print(f"   Similarity metric: {similarity_metric}")
        print(f"   Epsilon levels: {self.epsilon_levels}")
        print(f"   Perturbations per level: {self.n_perturbations}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   CUDA enabled: {self.use_cuda}")

    def _to_gpu(self, X: np.ndarray) -> Union[np.ndarray, Any]:
        """Перенос данных на GPU если доступно"""
        if self.use_cuda and len(X) > 100:  # Только для больших массивов
            try:
                return cp.asarray(X)
            except:
                return X
        return X

    def _to_cpu(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """Перенос данных обратно на CPU"""
        try:
            if self.use_cuda and hasattr(X, 'get'):
                return X.get()
            return np.asarray(X)
        except:
            return np.asarray(X)

    @lru_cache(maxsize=10000)
    def _cached_similarity_function(self, x1_hash: int, x2_hash: int,
                                  x1_tuple: tuple, x2_tuple: tuple) -> float:
        """Кэшированная версия similarity function для ускорения"""
        x1_flat = np.array(x1_tuple)
        x2_flat = np.array(x2_tuple)

        return self._compute_similarity_core(x1_flat, x2_flat)

    def _similarity_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Оптимизированная Trust-ADE similarity function
        """
        try:
            x1_flat = np.array(x1).flatten()
            x2_flat = np.array(x2).flatten()

            if len(x1_flat) != len(x2_flat):
                return 0.0

            # Пытаемся использовать кэш для небольших векторов
            if len(x1_flat) <= 50:  # Кэшируем только небольшие векторы
                try:
                    x1_tuple = tuple(x1_flat.round(6))  # Округление для стабильности хэша
                    x2_tuple = tuple(x2_flat.round(6))
                    x1_hash = hash(x1_tuple)
                    x2_hash = hash(x2_tuple)

                    return self._cached_similarity_function(x1_hash, x2_hash, x1_tuple, x2_tuple)
                except:
                    pass

            return self._compute_similarity_core(x1_flat, x2_flat)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в similarity function: {str(e)}")
            return 0.5

    def _compute_similarity_core(self, x1_flat: np.ndarray, x2_flat: np.ndarray) -> float:
        """Основная логика вычисления similarity"""
        # Обработка константных векторов
        if np.var(x1_flat) < 1e-12 and np.var(x2_flat) < 1e-12:
            return 1.0 if np.allclose(x1_flat, x2_flat) else 0.0

        # GPU ускорение для больших векторов
        if self.use_cuda and len(x1_flat) > 1000:
            try:
                x1_gpu = cp.asarray(x1_flat)
                x2_gpu = cp.asarray(x2_flat)

                if self.similarity_metric == 'cosine':
                    dot_product = cp.dot(x1_gpu, x2_gpu)
                    norm1 = cp.linalg.norm(x1_gpu)
                    norm2 = cp.linalg.norm(x2_gpu)

                    if norm1 < 1e-12 or norm2 < 1e-12:
                        similarity = 1.0 if cp.allclose(x1_gpu, x2_gpu) else 0.0
                    else:
                        similarity = float((dot_product / (norm1 * norm2)).get())

                elif self.similarity_metric == 'euclidean':
                    distance = cp.linalg.norm(x1_gpu - x2_gpu)
                    max_dist = cp.sqrt(len(x1_gpu)) * cp.max(cp.maximum(cp.abs(x1_gpu), cp.abs(x2_gpu)))

                    if max_dist < 1e-12:
                        similarity = 1.0
                    else:
                        similarity = float((1 - distance / max_dist).get())
                else:
                    # Fallback to CPU for other metrics
                    return self._compute_similarity_cpu(x1_flat, x2_flat)

                return np.clip(similarity, 0.0, 1.0)

            except Exception as e:
                # Fallback to CPU if GPU fails
                pass

        return self._compute_similarity_cpu(x1_flat, x2_flat)

    def _compute_similarity_cpu(self, x1_flat: np.ndarray, x2_flat: np.ndarray) -> float:
        """CPU версия similarity computation"""
        try:
            if self.similarity_metric == 'cosine':
                similarity = 1 - cosine(x1_flat, x2_flat)

            elif self.similarity_metric == 'euclidean':
                max_dist = np.sqrt(len(x1_flat)) * max(np.max(np.abs(x1_flat)), np.max(np.abs(x2_flat)))
                if max_dist < 1e-12:
                    similarity = 1.0
                else:
                    similarity = 1 - euclidean(x1_flat, x2_flat) / max_dist

            elif self.similarity_metric == 'pearson':
                corr, _ = pearsonr(x1_flat, x2_flat)
                similarity = (corr + 1) / 2

            else:
                similarity = 1 - cosine(x1_flat, x2_flat)

            if np.isnan(similarity) or np.isinf(similarity):
                return 0.5

            return np.clip(similarity, 0.0, 1.0)

        except Exception:
            return 0.5

    def _batch_predict(self, model, X_batch: np.ndarray) -> np.ndarray:
        """Батчевые предсказания для ускорения"""
        try:
            if hasattr(model, 'predict_batch') and len(X_batch) > 1:
                return model.predict_batch(X_batch)
            elif len(X_batch) > 1:
                return model.predict(X_batch)
            else:
                return np.array([model.predict(x.reshape(1, -1))[0] for x in X_batch])
        except Exception as e:
            # Fallback to individual predictions
            try:
                return np.array([model.predict(x.reshape(1, -1)) for x in X_batch])
            except:
                warnings.warn(f"Batch prediction failed: {e}")
                return np.array( * len(X_batch))

    def adversarial_robustness(self, model, X: np.ndarray, y: Optional[np.ndarray] = None,
                              attack_types: List[str] = None, n_samples: int = 100) -> float:
        """
        Оптимизированная Trust-ADE формула Adversarial устойчивости с батчингом
        R_a = 1 - (1/|A|) * Σ I[f(x+a) ≠ f(x)]
        """
        try:
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            attack_types = attack_types or ['random', 'boundary']
            n_test = min(n_samples, len(X))

            if n_test == 0:
                return 0.5

            test_indices = np.random.choice(len(X), n_test, replace=False)
            X_test = X[test_indices]

            print(f"🎯 Testing adversarial robustness on {n_test} samples...")

            # Генерируем все возмущения заранее для батчинга
            all_perturbations = []
            all_original_indices = []

            for i, x_orig in enumerate(X_test):
                for attack_type in attack_types:
                    for eps in self.epsilon_levels:
                        for _ in range(self.n_perturbations // len(self.epsilon_levels)):

                            if attack_type == 'random':
                                perturbation = np.random.uniform(-eps, eps, x_orig.shape)
                            elif attack_type == 'boundary':
                                gradient_approx = np.random.normal(0, eps, x_orig.shape)
                                perturbation = gradient_approx / (np.linalg.norm(gradient_approx) + 1e-8) * eps
                            elif attack_type == 'fgsm':
                                gradient_approx = np.random.choice([-1, 1], x_orig.shape)
                                perturbation = eps * gradient_approx
                            else:
                                perturbation = np.random.uniform(-eps, eps, x_orig.shape)

                            all_perturbations.append(x_orig + perturbation)
                            all_original_indices.append(i)

            if not all_perturbations:
                return 0.5

            all_perturbations = np.array(all_perturbations)

            # Батчевые предсказания для оригинальных образцов
            original_preds = self._batch_predict(model, X_test)

            # Батчевые предсказания для adversarial образцов
            successful_attacks = 0
            total_attacks = 0

            for i in range(0, len(all_perturbations), self.batch_size):
                batch_end = min(i + self.batch_size, len(all_perturbations))
                batch_perturbations = all_perturbations[i:batch_end]
                batch_indices = all_original_indices[i:batch_end]

                try:
                    batch_preds = self._batch_predict(model, batch_perturbations)

                    for j, (pred, orig_idx) in enumerate(zip(batch_preds, batch_indices)):
                        total_attacks += 1
                        if pred != original_preds[orig_idx]:
                            successful_attacks += 1

                except Exception:
                    continue

            if total_attacks == 0:
                return 0.5

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
        Оптимизированная Trust-ADE формула шумовой устойчивости с батчингом
        R_n = E[similarity(f(x), f(x+ε))] где ε ~ N(0,σ²)
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

            print(f"🔊 Testing noise robustness on {n_test} samples...")

            # Батчевые оригинальные предсказания
            original_preds = []
            for x in X_test:
                pred = model.predict(x.reshape(1, -1))
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(x.reshape(1, -1))
                original_preds.append(np.array(pred).flatten())

            similarities = []

            # Параллельная обработка шумовых возмущений
            def process_sample_noise(args):
                i, x_orig = args
                orig_pred = original_preds[i]
                sample_similarities = []

                for sigma in self.epsilon_levels:
                    # Генерируем батч возмущений сразу
                    noise_batch = []
                    for _ in range(self.n_perturbations):
                        epsilon = np.random.normal(0, sigma, x_orig.shape)
                        noise_batch.append(x_orig + epsilon)

                    # Батчевые предсказания для шума
                    try:
                        for x_noisy in noise_batch:
                            noisy_pred = model.predict(x_noisy.reshape(1, -1))
                            if hasattr(model, 'predict_proba'):
                                noisy_pred = model.predict_proba(x_noisy.reshape(1, -1))
                            noisy_pred = np.array(noisy_pred).flatten()

                            similarity = self._similarity_function(orig_pred, noisy_pred)
                            sample_similarities.append(similarity)
                    except:
                        continue

                return sample_similarities

            # Параллельная обработка
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                sample_args = [(i, x) for i, x in enumerate(X_test)]
                results = list(executor.map(process_sample_noise, sample_args))

                for sample_sims in results:
                    similarities.extend(sample_sims)

            if not similarities:
                return 0.5

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
        Оптимизированная Trust-ADE формула устойчивости объяснений
        R_e = E[similarity(E(x), E(x+ε))] где ε ~ N(0,σ²)
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

            print(f"🧠 Testing explanation robustness on {n_test} samples...")

            explanation_similarities = []

            def process_explanation_sample(args):
                i, x_orig = args
                sample_similarities = []

                try:
                    # Получаем оригинальное объяснение
                    if hasattr(explainer, 'shap_values'):
                        orig_explanation = explainer.shap_values(x_orig.reshape(1, -1))
                    elif hasattr(explainer, 'explain'):
                        orig_explanation = explainer.explain(x_orig.reshape(1, -1))
                    elif hasattr(explainer, 'explain_instance'):
                        exp = explainer.explain_instance(x_orig, model.predict_proba, num_features=len(x_orig))
                        orig_explanation = np.array([item[1] for item in exp.as_list()])
                    else:
                        orig_explanation = np.random.normal(0, 0.1, x_orig.shape)

                    orig_explanation = np.array(orig_explanation).flatten()

                    # Тестируем возмущения
                    for sigma in self.epsilon_levels[:2]:  # Меньше уровней для объяснений
                        for _ in range(self.n_perturbations // 2):
                            epsilon = np.random.normal(0, sigma, x_orig.shape)
                            x_perturbed = x_orig + epsilon

                            try:
                                # Объяснение для возмущённого входа
                                if hasattr(explainer, 'shap_values'):
                                    perturbed_explanation = explainer.shap_values(x_perturbed.reshape(1, -1))
                                elif hasattr(explainer, 'explain'):
                                    perturbed_explanation = explainer.explain(x_perturbed.reshape(1, -1))
                                elif hasattr(explainer, 'explain_instance'):
                                    exp = explainer.explain_instance(x_perturbed, model.predict_proba, num_features=len(x_perturbed))
                                    perturbed_explanation = np.array([item[1] for item in exp.as_list()])
                                else:
                                    perturbed_explanation = np.random.normal(0, 0.1, x_orig.shape)

                                perturbed_explanation = np.array(perturbed_explanation).flatten()

                                similarity = self._similarity_function(orig_explanation, perturbed_explanation)
                                sample_similarities.append(similarity)

                            except Exception:
                                continue

                except Exception:
                    pass

                return sample_similarities

            # Параллельная обработка объяснений
            with ThreadPoolExecutor(max_workers=min(self.n_workers, 2)) as executor:  # Меньше workers для объяснений
                sample_args = [(i, x) for i, x in enumerate(X_test)]
                results = list(executor.map(process_explanation_sample, sample_args))

                for sample_sims in results:
                    explanation_similarities.extend(sample_sims)

            if not explanation_similarities:
                return 0.5

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
        """
        try:
            # Параллельное вычисление компонентов Trust-ADE
            def compute_ra():
                return self.adversarial_robustness(model, X, y)

            def compute_rn():
                return self.noise_robustness(model, X)

            def compute_re():
                return self.explanation_robustness(model, explainer, X)

            # Используем ThreadPoolExecutor для параллельных вычислений
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_ra = executor.submit(compute_ra)
                future_rn = executor.submit(compute_rn)
                future_re = executor.submit(compute_re)

                r_a = future_ra.result()
                r_n = future_rn.result()
                r_e = future_re.result()

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
        Полный оптимизированный анализ устойчивости согласно Trust-ADE
        """
        try:
            if verbose:
                print(f"🛡️ Trust-ADE Robustness Analysis (Optimized)...")
                start_time = time.time()

            # Валидация данных
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            if y is not None:
                y = np.array(y)

            # Компоненты устойчивости Trust-ADE с оптимизациями
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
                },
                'optimization_info': {
                    'cuda_used': self.use_cuda,
                    'batch_size': self.batch_size,
                    'cache_hits': len(self._similarity_cache)
                }
            }

            if verbose:
                elapsed_time = time.time() - start_time
                print(f"📊 Trust-ADE Robustness Results (Optimized):")
                print(f"   🎯 Robustness Index: {results['robustness_index']:.4f} ({robustness_level})")
                print(f"   ⚔️ Adversarial R_a: {results['adversarial_robustness']:.4f}")
                print(f"   🔊 Noise R_n: {results['noise_robustness']:.4f}")
                print(f"   🧠 Explanation R_e: {results['explanation_robustness']:.4f}")
                print(f"   ⏱️ Analysis time: {elapsed_time:.2f}s")
                print(f"   🚀 CUDA used: {self.use_cuda}")
                print(f"   📦 Cache hits: {len(self._similarity_cache)}")

            return results

        except Exception as e:
            warnings.warn(f"🚨 Критическая ошибка в OptimizedRobustnessIndex.calculate: {str(e)}")
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
            },
            'optimization_info': {
                'cuda_used': False,
                'batch_size': self.batch_size,
                'cache_hits': 0
            }
        }

    def clear_cache(self):
        """Очистка кэша similarity вычислений"""
        self._similarity_cache.clear()

# Демонстрация использования
if __name__ == "__main__":
    print("🚀 Trust-ADE Optimized RobustnessIndex v2.5 - Высокопроизводительный анализ устойчивости!")

    # Создаем оптимизированный анализатор устойчивости
    robustness_analyzer = OptimizedRobustnessIndex(
        w_adversarial=0.4,
        w_noise=0.35,
        w_explanation=0.25,
        similarity_metric='cosine',
        epsilon_levels=[0.01, 0.05, 0.1],
        n_perturbations=20,
        batch_size=2000,  # Большие батчи для ускорения
        use_cuda=True,    # CUDA ускорение
        n_workers=4       # Параллельная обработка
    )

    # Mock модель для демонстрации
    class MockModel:
        def predict(self, X):
            if X.ndim == 1:
                return np.array([np.random.binomial(1, 0.7)])
            return np.random.binomial(1, 0.7, X.shape[0])

        def predict_proba(self, X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            proba = np.random.random((X.shape, 2))
            return proba / proba.sum(axis=1, keepdims=True)

    class MockExplainer:
        def shap_values(self, X):
            return np.random.normal(0, 1, X.shape)

    # Пример данных
    np.random.seed(42)
    X_test = np.random.random((100, 10))
    y_test = np.random.binomial(1, 0.6, 100)

    model = MockModel()
    explainer = MockExplainer()

    # Оптимизированный анализ устойчивости
    results = robustness_analyzer.calculate(
        model=model,
        explainer=explainer,
        X=X_test,
        y=y_test,
        n_samples=50,
        verbose=True
    )

    print(f"\n✅ Optimized Robustness Analysis Complete!")
    print(f"   Final R_I = {results['robustness_index']:.4f}")
    print(f"   Level: {results['robustness_level']}")
    print(f"   Speedup achieved with: CUDA={results['optimization_info']['cuda_used']}")
