"""
Основное демо Trust-ADE Protocol на медицинских данных
Демонстрирует полную функциональность системы оценки доверия
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import os

# Добавляем путь к проекту
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.sklearn_wrapper import SklearnWrapper
from trust_ade.trust_ade import TrustADE


def demo_medical_diagnosis():
    """Полная демонстрация Trust-ADE на медицинских данных"""
    print("=" * 60)
    print("🏥 ДЕМОНСТРАЦИЯ TRUST-ADE: МЕДИЦИНСКАЯ ДИАГНОСТИКА")
    print("=" * 60)

    try:
        # 1. Подготовка данных
        print("\n📊 Этап 1: Загрузка и подготовка данных")
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names

        print(f"✅ Загружено {X.shape[0]} образцов с {X.shape[1]} признаками")
        print(f"✅ Классы: {np.bincount(y)} (0=злокачественная, 1=доброкачественная)")

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Создаем референсные данные с небольшим дрейфом
        np.random.seed(42)
        X_reference = X_train + np.random.normal(0, 0.02, X_train.shape)
        y_reference = y_train

        print(f"✅ Обучающая выборка: {X_train.shape[0]} образцов")
        print(f"✅ Тестовая выборка: {X_test.shape[0]} образцов")
        print(f"✅ Референсная выборка: {X_reference.shape[0]} образцов")

        # 2. Обучение модели
        print("\n🤖 Этап 2: Обучение модели")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Оценка качества модели
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)

        print(f"✅ Модель обучена: Random Forest (100 деревьев)")
        print(f"✅ Точность на обучении: {train_accuracy:.3f}")
        print(f"✅ Точность на тесте: {test_accuracy:.3f}")

        # 3. Создание обертки модели
        wrapped_model = SklearnWrapper(model, feature_names=list(feature_names))
        print(f"✅ Модель обернута в SklearnWrapper")

        # 4. Создание защищенных атрибутов (симуляция возрастных групп)
        np.random.seed(42)
        protected_data = np.random.choice([0, 1], size=len(y_test), p=[0.6, 0.4])
        print(f"✅ Созданы защищенные атрибуты: {np.bincount(protected_data)} (0=молодые, 1=пожилые)")

        # 5. Инициализация Trust-ADE
        print("\n🎯 Этап 3: Инициализация Trust-ADE")
        trust_evaluator = TrustADE(
            model=wrapped_model,
            domain='medical',  # Медицинский домен - приоритет объяснимости
            protected_attributes=['age_group'],
            explainer_type='shap',
            training_data=X_train
        )

        print(f"✅ Trust-ADE инициализирован для домена: medical")
        print(f"✅ Объяснитель: SHAP TreeExplainer")
        print(f"✅ Защищенные атрибуты: {trust_evaluator.protected_attributes}")

        # 6. Выполнение полной оценки
        print("\n🔍 Этап 4: Выполнение комплексной оценки доверия")
        results = trust_evaluator.evaluate(
            X_test=X_test,
            y_test=y_test,
            X_reference=X_reference,
            y_reference=y_reference,
            protected_data=protected_data,
            verbose=True
        )

        # 7. Вывод основных результатов
        print("\n" + "=" * 60)
        print("📊 ОСНОВНЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)

        print(f"🎯 Trust Score: {results['trust_score']:.3f}")
        print(f"📈 Уровень доверия: {results['trust_level']}")
        print(f"🏷️ Домен: {results['domain']}")

        print(f"\n📋 Детализация компонентов:")
        print(f"  • Explainability Score: {results['explainability_score']:.3f}")
        print(f"  • Robustness Index: {results['robustness_index']:.3f}")
        print(f"  • Bias Shift Index: {results['bias_shift_index']:.3f}")
        print(f"  • Concept Drift Rate: {results['concept_drift_rate']:.3f}")

        # 8. Генерация и вывод полного отчета
        print("\n📄 Этап 5: Генерация подробного отчета")
        report = trust_evaluator.generate_report()
        print(report)

        # 9. Рекомендации
        print("\n💡 РЕКОМЕНДАЦИИ:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")

        print("\n✅ Демонстрация медицинской диагностики успешно завершена!")
        return results

    except Exception as e:
        print(f"\n❌ Ошибка в демонстрации: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def demo_financial_scoring():
    """Демонстрация Trust-ADE на финансовых данных"""
    print("\n" + "=" * 60)
    print("💳 ДЕМОНСТРАЦИЯ TRUST-ADE: КРЕДИТНЫЙ СКОРИНГ")
    print("=" * 60)

    try:
        from sklearn.datasets import make_classification

        # Создание синтетических финансовых данных
        print("\n📊 Создание синтетических финансовых данных...")
        X, y = make_classification(
            n_samples=2000,
            n_features=15,
            n_informative=12,
            n_redundant=3,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=42
        )

        feature_names = [
            'income', 'credit_history', 'debt_ratio', 'employment_length',
            'loan_amount', 'loan_term', 'credit_score', 'assets_value',
            'education', 'marital_status', 'dependents', 'residence_type',
            'job_type', 'bank_relationship', 'previous_loans'
        ]

        print(f"✅ Сгенерировано {X.shape[0]} заявок на кредит")
        print(f"✅ Признаки: {len(feature_names)} финансовых показателей")

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Нормализация данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Обучение модели
        print("\n🤖 Обучение модели логистической регрессии...")
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
        model.fit(X_train_scaled, y_train)

        accuracy = model.score(X_test_scaled, y_test)
        print(f"✅ Точность модели: {accuracy:.3f}")

        # Создание защищенных атрибутов (пол и возраст)
        np.random.seed(42)
        protected_data = np.random.choice([0, 1], size=len(y_test), p=[0.55, 0.45])
        print(f"✅ Защищенные атрибуты: {np.bincount(protected_data)} (0=мужчины, 1=женщины)")

        # Trust-ADE для финансового домена
        wrapped_model = SklearnWrapper(model, feature_names=feature_names)
        trust_evaluator = TrustADE(
            model=wrapped_model,
            domain='finance',  # Финансовый домен - баланс всех факторов
            protected_attributes=['gender'],
            training_data=X_train_scaled
        )

        print("\n🔍 Выполнение оценки Trust-ADE для финансовой модели...")
        results = trust_evaluator.evaluate(
            X_test=X_test_scaled,
            y_test=y_test,
            protected_data=protected_data,
            verbose=False
        )

        # Результаты
        print(f"\n🎯 Trust Score: {results['trust_score']:.3f}")
        print(f"📈 Уровень доверия: {results['trust_level']}")
        print(f"📊 Компоненты:")
        print(f"  • Объяснимость: {results['explainability_score']:.3f}")
        print(f"  • Устойчивость: {results['robustness_index']:.3f}")
        print(f"  • Справедливость: {1 - results['bias_shift_index']:.3f}")

        return results

    except Exception as e:
        print(f"❌ Ошибка в финансовой демонстрации: {str(e)}")
        return None


if __name__ == "__main__":
    print("🚀 Запуск демонстраций Trust-ADE Protocol")

    # Основная медицинская демонстрация
    medical_results = demo_medical_diagnosis()

    # Дополнительная финансовая демонстрация
    financial_results = demo_financial_scoring()

    # Сравнительный анализ
    if medical_results and financial_results:
        print("\n" + "=" * 60)
        print("🔍 СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
        print("=" * 60)

        print(f"Медицина  - Trust Score: {medical_results['trust_score']:.3f} ({medical_results['trust_level']})")
        print(f"Финансы   - Trust Score: {financial_results['trust_score']:.3f} ({financial_results['trust_level']})")

        print(f"\nРазличия в весах:")
        med_weights = medical_results['weights_used']
        fin_weights = financial_results['weights_used']

        print(f"Объяснимость: Медицина {med_weights['w_E']:.2f} vs Финансы {fin_weights['w_E']:.2f}")
        print(f"Устойчивость: Медицина {med_weights['w_R']:.2f} vs Финансы {fin_weights['w_R']:.2f}")
        print(f"Справедливость: Медицина {med_weights['w_F']:.2f} vs Финансы {fin_weights['w_F']:.2f}")

    print("\n🎉 Все демонстрации завершены!")
