"""
Trust-ADE Explainability Score Module
Реализация каузальной валидации объяснений согласно статье XAI 2.0
"""

import numpy as np
import warnings
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import entropy, ks_2samp
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Optional, Union, Any


class ExplainabilityScore:
    """
    Каузально-ориентированный Explainability Score для Trust-ADE протокола

    Реализует формулу: ES = w_c·F_c + w_s·C_s + w_i·S_i + w_h·U_h
    где:
    - F_c: каузальная фиделити с экспертными графами
    - C_s: семантическая когерентность через информационную энтропию
    - S_i: стабильность интерпретаций при ε-возмущениях
    - U_h: человеческая понятность через эксперименты
    """

    def __init__(self, causal_weight: float = 0.35, coherence_weight: float = 0.25,
                 stability_weight: float = 0.25, human_weight: float = 0.15,
                 alpha: float = 0.5, gamma: float = 1.0, noise_threshold: float = 1e-6):
        """
        Args:
            causal_weight: w_c - вес каузальной фиделити
            coherence_weight: w_s - вес семантической когерентности
            stability_weight: w_i - вес стабильности интерпретаций
            human_weight: w_h - вес человеческой понятности
            alpha: параметр балансировки полноты и точности в F_c
            gamma: чувствительность к concept drift
            noise_threshold: порог для фильтрации шума
        """
        # Нормализация весов до 1.0
        total_weight = causal_weight + coherence_weight + stability_weight + human_weight

        self.w_c = causal_weight / total_weight
        self.w_s = coherence_weight / total_weight
        self.w_i = stability_weight / total_weight
        self.w_h = human_weight / total_weight

        self.alpha = alpha  # для формулы каузальной фиделити
        self.gamma = gamma  # для concept drift
        self.noise_threshold = noise_threshold

        print(f"🧠 Trust-ADE Explainability Score initialized:")
        print(f"   Causal Fidelity weight: {self.w_c:.3f}")
        print(f"   Semantic Coherence weight: {self.w_s:.3f}")
        print(f"   Interpretation Stability weight: {self.w_i:.3f}")
        print(f"   Human Comprehensibility weight: {self.w_h:.3f}")

    def causal_fidelity(self, system_edges: set, expert_edges: set,
                       confidence_scores: Optional[Dict] = None,
                       snr_ratio: Optional[float] = None) -> float:
        """
        Каузальная фиделити согласно формуле из статьи:
        F_c = |E_sys ∩ E_exp|/|E_exp| × α + |E_sys ∩ E_exp|/|E_sys| × (1-α)

        С робастной модификацией для зашумленных данных:
        F_c_robust = F_c × (1 - η·SNR^(-1))

        Args:
            system_edges: каузальные связи, выявленные системой
            expert_edges: экспертные каузальные связи
            confidence_scores: уровень консенсуса экспертов
            snr_ratio: отношение сигнал/шум для робастности

        Returns:
            float: каузальная фиделити [0, 1]
        """
        try:
            if len(expert_edges) == 0 or len(system_edges) == 0:
                return 0.5  # нейтральная оценка при отсутствии данных

            # Пересечение каузальных связей
            intersection = system_edges.intersection(expert_edges)

            # Базовая формула F_c из статьи
            recall = len(intersection) / len(expert_edges)  # полнота
            precision = len(intersection) / len(system_edges)  # точность

            base_fidelity = self.alpha * recall + (1 - self.alpha) * precision

            # Робастная модификация для конфликтных экспертных мнений
            if confidence_scores is not None:
                # Консервативная оценка для спорных связей
                disputed_edges = {edge for edge in expert_edges
                                if confidence_scores.get(edge, 1.0) < 0.7}

                # Исключаем спорные связи из базовой оценки
                clean_expert = expert_edges - disputed_edges
                clean_intersection = system_edges.intersection(clean_expert)

                if len(clean_expert) > 0:
                    clean_recall = len(clean_intersection) / len(clean_expert)
                    clean_precision = len(clean_intersection) / len(system_edges)
                    base_fidelity = self.alpha * clean_recall + (1 - self.alpha) * clean_precision

            # Модификация для зашумленных данных
            if snr_ratio is not None and snr_ratio > 0:
                eta = 0.1  # коэффициент влияния шума
                noise_penalty = eta / snr_ratio
                robust_fidelity = base_fidelity * (1 - noise_penalty)
                return np.clip(robust_fidelity, 0.0, 1.0)

            return np.clip(base_fidelity, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в causal_fidelity: {str(e)}")
            return 0.5

    def extract_causal_edges_from_explanations(self, explanations: np.ndarray,
                                             feature_names: Optional[List[str]] = None,
                                             threshold: float = 0.1) -> set:
        """
        Извлечение каузальных связей из матрицы объяснений

        Args:
            explanations: матрица важности признаков [n_samples, n_features]
            feature_names: имена признаков
            threshold: порог значимости для каузальной связи

        Returns:
            set: множество каузальных связей (пар признаков)
        """
        try:
            if explanations is None or len(explanations) == 0:
                return set()

            explanations = np.array(explanations)
            n_samples, n_features = explanations.shape

            # Каузальные связи через взаимную информацию
            causal_edges = set()

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # Взаимная информация между признаками i и j
                    try:
                        mi_score = mutual_info_score(
                            explanations[:, i] > threshold,
                            explanations[:, j] > threshold
                        )

                        # Если взаимная информация выше порога - считаем каузальной связью
                        if mi_score > 0.1:
                            if feature_names:
                                edge = (feature_names[i], feature_names[j])
                            else:
                                edge = (i, j)
                            causal_edges.add(edge)

                    except Exception:
                        continue

            return causal_edges

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в extract_causal_edges: {str(e)}")
            return set()

    def semantic_coherence(self, explanations: np.ndarray) -> float:
        """
        Семантическая когерентность согласно формуле:
        C_s = 1 - H(E)/H_max
        где H(E) - энтропия распределения объяснений

        Args:
            explanations: матрица объяснений

        Returns:
            float: семантическая когерентность [0, 1]
        """
        try:
            if explanations is None or len(explanations) == 0:
                return 0.0

            # Нормализация объяснений в распределение вероятностей
            exp_flat = np.abs(explanations).flatten()
            exp_filtered = exp_flat[exp_flat > self.noise_threshold]

            if len(exp_filtered) <= 1:
                return 1.0  # максимальная когерентность для константы

            # Нормализация в вероятностное распределение
            total_mass = np.sum(exp_filtered)
            if total_mass < 1e-12:
                return 0.5

            prob_dist = exp_filtered / total_mass

            # Информационная энтропия H(E)
            H_E = entropy(prob_dist, base=2)
            H_max = np.log2(len(prob_dist))  # максимальная энтропия

            # C_s = 1 - H(E)/H_max (больше когерентности = меньше энтропии)
            coherence = 1.0 - (H_E / H_max) if H_max > 0 else 1.0

            return np.clip(coherence, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в semantic_coherence: {str(e)}")
            return 0.5

    def interpretation_stability(self, model: Any, explainer: Any, X: np.ndarray,
                               perturbation_sizes: List[float] = [0.01, 0.05, 0.1],
                               n_samples: int = 25, distance_metric: str = 'cosine') -> float:
        """
        Стабильность интерпретаций согласно формуле:
        S_i = 1 - (1/N) ∑ d(E_i, E_i^ε)
        где d - метрика расстояния между объяснениями

        Args:
            model: модель для предсказаний
            explainer: объяснитель
            X: входные данные
            perturbation_sizes: размеры возмущений ε
            n_samples: количество тестовых образцов
            distance_metric: метрика расстояния ('cosine', 'euclidean', 'manhattan')

        Returns:
            float: стабильность интерпретаций [0, 1]
        """
        try:
            if X is None or len(X) == 0:
                return 0.0

            X = np.array(X)
            n_test = min(n_samples, len(X))
            all_stabilities = []

            # Тестируем стабильность для разных размеров возмущений
            for eps in perturbation_sizes:
                eps_stabilities = []

                for i in range(n_test):
                    try:
                        # Оригинальный образец
                        x_orig = X[i:i+1]

                        # Возмущенный образец
                        noise = np.random.normal(0, eps, x_orig.shape)
                        x_pert = x_orig + noise

                        # Получаем объяснения
                        exp_orig = self._safe_explain(explainer, x_orig)
                        exp_pert = self._safe_explain(explainer, x_pert)

                        if exp_orig is not None and exp_pert is not None:
                            # Вычисляем расстояние между объяснениями
                            distance = self._compute_explanation_distance(
                                exp_orig, exp_pert, distance_metric
                            )

                            # S_i = 1 - distance (больше сходства = больше стабильности)
                            stability = max(0.0, 1.0 - distance)
                            eps_stabilities.append(stability)

                    except Exception:
                        continue  # пропускаем проблемные образцы

                if eps_stabilities:
                    all_stabilities.extend(eps_stabilities)

            return np.mean(all_stabilities) if all_stabilities else 0.5

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в interpretation_stability: {str(e)}")
            return 0.5

    def _safe_explain(self, explainer: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Безопасное получение объяснений с обработкой ошибок"""
        try:
            if hasattr(explainer, 'explain_instance'):
                # LIME-style explainer
                explanations = []
                for sample in X:
                    exp = explainer.explain_instance(sample, explainer.predict_fn)
                    if hasattr(exp, 'as_list'):
                        exp_values = [val for _, val in exp.as_list()]
                        explanations.append(exp_values)
                return np.array(explanations) if explanations else None

            elif hasattr(explainer, 'shap_values'):
                # SHAP explainer
                return explainer.shap_values(X)

            elif hasattr(explainer, '__call__'):
                # Generic callable explainer
                return explainer(X)

            else:
                return None

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в _safe_explain: {str(e)}")
            return None

    def _compute_explanation_distance(self, exp1: np.ndarray, exp2: np.ndarray,
                                    metric: str = 'cosine') -> float:
        """Вычисление расстояния между объяснениями"""
        try:
            exp1_flat = np.array(exp1).flatten()
            exp2_flat = np.array(exp2).flatten()

            if len(exp1_flat) != len(exp2_flat):
                # Приводим к одинаковой длине
                min_len = min(len(exp1_flat), len(exp2_flat))
                exp1_flat = exp1_flat[:min_len]
                exp2_flat = exp2_flat[:min_len]

            if metric == 'cosine':
                return cosine(exp1_flat, exp2_flat)
            elif metric == 'euclidean':
                return euclidean(exp1_flat, exp2_flat) / np.sqrt(len(exp1_flat))
            elif metric == 'manhattan':
                return np.mean(np.abs(exp1_flat - exp2_flat))
            else:
                return cosine(exp1_flat, exp2_flat)  # default

        except Exception:
            return 1.0  # максимальное расстояние при ошибке

    def human_comprehensibility(self, explanations: np.ndarray,
                              expert_ratings: Optional[List[float]] = None,
                              complexity_factors: Optional[Dict] = None) -> float:
        """
        Человеческая понятность через контролируемые эксперименты

        Args:
            explanations: матрица объяснений
            expert_ratings: экспертные оценки по стандартизированной шкале [0, 1]
            complexity_factors: факторы сложности для эвристической оценки

        Returns:
            float: человеческая понятность [0, 1]
        """
        try:
            # Если есть экспертные оценки - используем их
            if expert_ratings is not None and len(expert_ratings) > 0:
                return np.clip(np.mean(expert_ratings), 0.0, 1.0)

            # Эвристическая оценка через принципы когнитивной нагрузки
            if explanations is None or len(explanations) == 0:
                return 0.0

            explanations = np.array(explanations)

            # 1. Разреженность (спарсность) - меньше активных признаков = лучше
            sparsity_scores = []
            for exp in explanations:
                non_zero_ratio = np.sum(np.abs(exp) > self.noise_threshold) / len(exp)
                # Оптимальная спарсность: 10-20% активных признаков
                optimal_sparsity = 0.15
                sparsity_penalty = abs(non_zero_ratio - optimal_sparsity) / optimal_sparsity
                sparsity_score = max(0.0, 1.0 - sparsity_penalty)
                sparsity_scores.append(sparsity_score)

            sparsity_metric = np.mean(sparsity_scores)

            # 2. Консистентность важности признаков
            feature_importance_var = np.var(np.mean(np.abs(explanations), axis=0))
            consistency_metric = 1.0 / (1.0 + feature_importance_var)

            # 3. Мономодальность распределения важности (один ясный пик лучше)
            mean_importance = np.mean(np.abs(explanations), axis=0)
            max_importance = np.max(mean_importance)
            second_max = np.partition(mean_importance, -2)[-2] if len(mean_importance) > 1 else 0

            dominance_ratio = max_importance / (second_max + 1e-8)
            dominance_metric = np.tanh(dominance_ratio - 1.0)  # оптимально 2-3x разница

            # Комбинированная эвристическая оценка
            w_sparsity = 0.4
            w_consistency = 0.3
            w_dominance = 0.3

            heuristic_score = (w_sparsity * sparsity_metric +
                             w_consistency * consistency_metric +
                             w_dominance * dominance_metric)

            return np.clip(heuristic_score, 0.0, 1.0)

        except Exception as e:
            warnings.warn(f"🚨 Ошибка в human_comprehensibility: {str(e)}")
            return 0.5

    def calculate(self, model: Any, explainer: Any, X: np.ndarray, y: Optional[np.ndarray] = None,
                 expert_graph: Optional[Dict] = None, expert_ratings: Optional[List[float]] = None,
                 feature_names: Optional[List[str]] = None,
                 verbose: bool = True) -> Dict[str, float]:
        """
        Вычисление итогового Explainability Score согласно Trust-ADE

        ES = w_c·F_c + w_s·C_s + w_i·S_i + w_h·U_h

        Args:
            model: модель для предсказаний
            explainer: объяснитель (LIME, SHAP, etc.)
            X: входные данные
            y: целевые переменные (опционально)
            expert_graph: экспертный каузальный граф
            expert_ratings: экспертные оценки понятности
            feature_names: имена признаков
            verbose: детальный вывод результатов

        Returns:
            Dict[str, float]: результаты всех компонент ES
        """
        try:
            if X is None or len(X) == 0:
                return self._default_results()

            X = np.array(X)

            # Ограничиваем размер выборки для производительности
            n_samples = min(100, len(X))  # увеличили для лучшей точности
            if n_samples < len(X):
                sample_indices = np.random.choice(len(X), n_samples, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X

            if verbose:
                print(f"🔍 Trust-ADE анализ на {len(X_sample)} образцах...")

            # 1. Получаем объяснения
            explanations = self._safe_explain(explainer, X_sample)
            if explanations is None or len(explanations) == 0:
                warnings.warn("❌ Не удалось получить объяснения")
                return self._default_results()

            explanations = np.array(explanations)

            # 2. Извлекаем каузальные связи из объяснений
            system_edges = self.extract_causal_edges_from_explanations(
                explanations, feature_names
            )

            # 3. Каузальная фиделити F_c
            if expert_graph and 'causal_edges' in expert_graph:
                expert_edges = set(expert_graph['causal_edges'])
                confidence_scores = expert_graph.get('confidence_scores')
                snr_ratio = expert_graph.get('snr_ratio')

                F_c = self.causal_fidelity(system_edges, expert_edges,
                                         confidence_scores, snr_ratio)
            else:
                # Эвристическая оценка через консистентность
                F_c = self._heuristic_causal_consistency(explanations)

            # 4. Семантическая когерентность C_s
            C_s = self.semantic_coherence(explanations)

            # 5. Стабильность интерпретаций S_i
            S_i = self.interpretation_stability(model, explainer, X_sample)

            # 6. Человеческая понятность U_h
            U_h = self.human_comprehensibility(explanations, expert_ratings)

            # 7. Итоговый Explainability Score
            ES = (self.w_c * F_c + self.w_s * C_s +
                  self.w_i * S_i + self.w_h * U_h)

            results = {
                'explainability_score': np.clip(ES, 0.0, 1.0),
                'causal_fidelity': np.clip(F_c, 0.0, 1.0),
                'semantic_coherence': np.clip(C_s, 0.0, 1.0),
                'interpretation_stability': np.clip(S_i, 0.0, 1.0),
                'human_comprehensibility': np.clip(U_h, 0.0, 1.0),
                'n_causal_edges': len(system_edges),
                'n_expert_edges': len(expert_graph.get('causal_edges', [])) if expert_graph else 0
            }

            if verbose:
                print(f"📊 Trust-ADE Explainability Score Results:")
                print(f"   🧠 Explainability Score: {results['explainability_score']:.4f}")
                print(f"   🔗 Causal Fidelity: {results['causal_fidelity']:.4f}")
                print(f"   🧩 Semantic Coherence: {results['semantic_coherence']:.4f}")
                print(f"   ⚖️ Interpretation Stability: {results['interpretation_stability']:.4f}")
                print(f"   👥 Human Comprehensibility: {results['human_comprehensibility']:.4f}")
                print(f"   📈 Detected Causal Edges: {results['n_causal_edges']}")

            return results

        except Exception as e:
            warnings.warn(f"🚨 Критическая ошибка в ExplainabilityScore.calculate: {str(e)}")
            return self._default_results()

    def _heuristic_causal_consistency(self, explanations: np.ndarray) -> float:
        """Эвристическая оценка каузальной консистентности через ранговые корреляции"""
        try:
            if len(explanations) < 2:
                return 0.5

            # Ранжируем признаки по важности для каждого объяснения
            rank_correlations = []

            for i in range(min(20, len(explanations))):  # ограничиваем для производительности
                for j in range(i + 1, min(20, len(explanations))):
                    try:
                        # Ранги важности признаков
                        rank_i = np.argsort(np.abs(explanations[i]))[::-1]
                        rank_j = np.argsort(np.abs(explanations[j]))[::-1]

                        # Spearman correlation между рангами
                        corr = np.corrcoef(rank_i, rank_j)[0, 1]
                        if not np.isnan(corr) and not np.isinf(corr):
                            rank_correlations.append(abs(corr))
                    except Exception:
                        continue

            return np.mean(rank_correlations) if rank_correlations else 0.5

        except Exception:
            return 0.5

    def _default_results(self) -> Dict[str, float]:
        """Результаты по умолчанию при критических ошибках"""
        return {
            'explainability_score': 0.5,
            'causal_fidelity': 0.5,
            'semantic_coherence': 0.5,
            'interpretation_stability': 0.5,
            'human_comprehensibility': 0.5,
            'n_causal_edges': 0,
            'n_expert_edges': 0
        }