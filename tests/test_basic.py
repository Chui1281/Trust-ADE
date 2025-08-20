"""
Сравнение различных ML моделей с помощью Trust-ADE Protocol
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Добавляем путь к проекту
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.sklearn_wrapper import SklearnWrapper
from trust_ade.trust_ade import TrustADE


def compare_models():
    """Сравнение различных моделей по Trust-ADE метрикам"""
    print("=" * 70)
    print("🔬 СРАВНЕНИЕ МОДЕЛЕЙ ML С ПОМОЩЬЮ TRUST-ADE")
    print("=" * 70)

    # Создание сбалансированного датасета
    print("\n📊 Создание тестового датасета...")
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"✅ Обучающая выборка: {X_train.shape}")
    print(f"✅ Тестовая выборка: {X_test.shape}")

    # Определение моделей для сравнения
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
    }

    results = {}
    trust_scores = {}

    print(f"\n🤖 Обучение и оценка {len(models)} моделей...")
    print("-" * 70)

    for name, model in models.items():
        print(f"\n📈 Модель: {name}")

        try:
            # Обучение модели
            print("  🔄 Обучение...")
            model.fit(X_train, y_train)

            # Оценка точности
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)

            print(f"  ✅ Точность на обучении: {train_acc:.3f}")
            print(f"  ✅ Точность на тесте: {test_acc:.3f}")

            # Создание wrapper
            wrapped_model = SklearnWrapper(model)

            # Trust-ADE оценка
            print("  🔍 Оценка Trust-ADE...")
            trust_evaluator = TrustADE(
                model=wrapped_model,
                domain='general',
                training_data=X_train
            )

            trust_result = trust_evaluator.evaluate(
                X_test=X_test,
                y_test=y_test,
                verbose=False
            )

            # Сохранение результатов
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'trust_score': trust_result['trust_score'],
                'trust_level': trust_result['trust_level'],
                'explainability': trust_result['explainability_score'],
                'robustness': trust_result['robustness_index'],
                'bias_shift': trust_result['bias_shift_index'],
                'concept_drift': trust_result['concept_drift_rate']
            }

            trust_scores[name] = trust_result['trust_score']

            print(f"  🎯 Trust Score: {trust_result['trust_score']:.3f}")
            print(f"  📊 Уровень доверия: {trust_result['trust_level']}")

        except Exception as e:
            print(f"  ❌ Ошибка для {name}: {str(e)}")
            continue

    # Анализ результатов
    print("\n" + "=" * 70)
    print("📊 СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    if not results:
        print("❌ Нет результатов для анализа")
        return

    # Таблица результатов
    print(f"\n{'Модель':<20} {'Точность':<10} {'Trust':<8} {'Уровень доверия':<20}")
    print("-" * 65)

    for name, result in results.items():
        print(f"{name:<20} {result['test_accuracy']:<10.3f} "
              f"{result['trust_score']:<8.3f} {result['trust_level']:<20}")

    # Детальная разбивка по компонентам
    print(f"\n{'Модель':<20} {'Объясн.':<8} {'Устойч.':<8} {'Смещ.':<8} {'Дрейф':<8}")
    print("-" * 60)

    for name, result in results.items():
        print(f"{name:<20} {result['explainability']:<8.3f} "
              f"{result['robustness']:<8.3f} {result['bias_shift']:<8.3f} "
              f"{result['concept_drift']:<8.3f}")

    # Рейтинг моделей
    print(f"\n🏆 РЕЙТИНГ ПО TRUST SCORE:")
    sorted_models = sorted(trust_scores.items(), key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(sorted_models, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"  {medal} {name}: {score:.3f}")

    # Выводы и рекомендации
    print(f"\n💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ:")

    best_model = sorted_models[0][0]
    best_score = sorted_models[1]

    print(f"  • Лучшая модель по Trust-ADE: {best_model} ({best_score:.3f})")

    # Анализ по компонентам
    best_explainability = max(results.items(), key=lambda x: x[1]['explainability'])
    best_robustness = max(results.items(), key=lambda x: x[1]['robustness'])

    print(f"  • Лучшая объяснимость: {best_explainability[0]} ({best_explainability[1]['explainability']:.3f})")
    print(f"  • Лучшая устойчивость: {best_robustness[0]} ({best_robustness[1]['robustness']:.3f})")

    # Предупреждения
    low_trust_models = [name for name, score in trust_scores.items() if score < 0.5]
    if low_trust_models:
        print(f"  ⚠️  Модели с низким доверием: {', '.join(low_trust_models)}")

    return results


def analyze_domain_sensitivity():
    """Анализ чувствительности к различным доменам"""
    print("\n" + "=" * 70)
    print("🏷️ АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ К ДОМЕНАМ")
    print("=" * 70)

    # Простой датасет для быстрого тестирования
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Одна модель - разные домены
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)
    wrapped_model = SklearnWrapper(model)

    domains = ['general', 'medical', 'finance', 'criminal_justice']
    domain_results = {}

    print(f"\n🔍 Тестирование модели в {len(domains)} доменах...")

    for domain in domains:
        print(f"\n📋 Домен: {domain}")

        trust_evaluator = TrustADE(
            model=wrapped_model,
            domain=domain,
            training_data=X_train
        )

        result = trust_evaluator.evaluate(X_test, y_test, verbose=False)
        domain_results[domain] = result

        print(f"  🎯 Trust Score: {result['trust_score']:.3f}")

        weights = result['weights_used']
        print(f"  ⚖️ Веса: E={weights['w_E']:.2f}, R={weights['w_R']:.2f}, F={weights['w_F']:.2f}")

    # Сравнение доменов
    print(f"\n📊 Сравнение доменов:")
    print(f"{'Домен':<20} {'Trust Score':<12} {'Приоритет'}")
    print("-" * 50)

    domain_priorities = {
        'general': 'Сбалансированный',
        'medical': 'Объяснимость',
        'finance': 'Устойчивость',
        'criminal_justice': 'Справедливость'
    }

    for domain, result in domain_results.items():
        priority = domain_priorities.get(domain, 'Неизвестно')
        print(f"{domain:<20} {result['trust_score']:<12.3f} {priority}")

    return domain_results


if __name__ == "__main__":
    print("🚀 Запуск сравнительного анализа моделей")

    # Основное сравнение моделей
    model_results = compare_models()

    # Анализ доменной чувствительности
    domain_results = analyze_domain_sensitivity()

    print(f"\n🎉 Сравнительный анализ завершен!")
    print(f"📊 Проанализировано моделей: {len(model_results) if model_results else 0}")
    print(f"🏷️ Протестировано доменов: {len(domain_results) if domain_results else 0}")
