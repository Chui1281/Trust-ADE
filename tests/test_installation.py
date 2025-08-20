"""
Скрипт для тестирования корректности установки Trust-ADE Protocol
"""

import sys
import os
import warnings


def test_basic_imports():
    """Тест базовых импортов"""
    print("🔍 Тестирование импортов...")

    try:
        # Добавляем путь к проекту
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)

        # Основные импорты
        from trust_ade import TrustADE
        from trust_ade import ExplainabilityScore, RobustnessIndex, BiasShiftIndex
        from models import SklearnWrapper
        from explainers import SHAPExplainer

        print("✅ Все основные модули импортированы успешно")
        return True

    except ImportError as e:
        print(f"❌ Ошибка импорта: {str(e)}")
        return False


def test_dependencies():
    """Тест зависимостей"""
    print("\n📦 Тестирование зависимостей...")

    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'scipy'
    ]

    optional_packages = [
        'shap', 'matplotlib', 'seaborn'
    ]

    missing_required = []
    missing_optional = []

    # Проверка обязательных пакетов
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package} (обязательный)")

    # Проверка опциональных пакетов
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} (опциональный)")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️ {package} (опциональный)")

    if missing_required:
        print(f"\n❌ Отсутствуют обязательные пакеты: {', '.join(missing_required)}")
        return False

    if missing_optional:
        print(f"\n⚠️ Отсутствуют опциональные пакеты: {', '.join(missing_optional)}")
        print("Некоторые функции могут быть недоступны")

    return True


def test_basic_functionality():
    """Тест базовой функциональности"""
    print("\n⚙️ Тестирование базовой функциональности...")

    try:
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        # Добавляем путь к проекту
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)

        from models.sklearn_wrapper import SklearnWrapper
        from trust_ade import TrustADE

        # Создание простых тестовых данных
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Простая модель
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Wrapper
        wrapped_model = SklearnWrapper(model)
        print("✅ SklearnWrapper создан")

        # Trust-ADE
        trust_evaluator = TrustADE(wrapped_model, domain='general', training_data=X)
        print("✅ TrustADE инициализирован")

        # Простая оценка
        results = trust_evaluator.evaluate(X[:20], y[:20], verbose=False)
        trust_score = results['trust_score']

        print(f"✅ Оценка выполнена, Trust Score: {trust_score:.3f}")

        if 0 <= trust_score <= 1:
            print("✅ Trust Score в корректном диапазоне")
            return True
        else:
            print(f"❌ Trust Score вне диапазона [0,1]: {trust_score}")
            return False

    except Exception as e:
        print(f"❌ Ошибка в тестировании функциональности: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_explainer_compatibility():
    """Тест совместимости с различными explainer"""
    print("\n🔍 Тестирование совместимости explainer...")

    try:
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier

        # Добавляем путь к проекту
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)

        from models.sklearn_wrapper import SklearnWrapper
        from trust_ade import TrustADE
        from trust_ade.utils import safe_explain

        # Тестовые данные
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        wrapped_model = SklearnWrapper(model)

        # Тест с SHAP
        try:
            trust_evaluator = TrustADE(wrapped_model, explainer_type='shap', training_data=X)
            explanations = safe_explain(trust_evaluator.explainer, X[:5])

            if explanations.shape == (5, 5):
                print("✅ SHAP explainer работает корректно")
            else:
                print(f"⚠️ SHAP explainer: неожиданная форма {explanations.shape}")

        except Exception as e:
            print(f"⚠️ SHAP explainer: {str(e)}")

        return True

    except Exception as e:
        print(f"❌ Ошибка в тестировании explainer: {str(e)}")
        return False


def generate_installation_report():
    """Генерация отчета об установке"""
    print("\n" + "=" * 60)
    print("📋 ОТЧЕТ ОБ УСТАНОВКЕ TRUST-ADE PROTOCOL")
    print("=" * 60)

    tests = [
        ("Импорты", test_basic_imports),
        ("Зависимости", test_dependencies),
        ("Функциональность", test_basic_functionality),
        ("Explainer совместимость", test_explainer_compatibility)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}:")
        print("-" * 40)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = test_func()

            if result:
                print(f"✅ {test_name}: ПРОЙДЕН")
                passed_tests += 1
            else:
                print(f"❌ {test_name}: НЕ ПРОЙДЕН")

        except Exception as e:
            print(f"💥 {test_name}: КРИТИЧЕСКАЯ ОШИБКА - {str(e)}")

    # Итоги
    print(f"\n" + "=" * 60)
    print(f"📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print(f"=" * 60)

    print(f"Пройдено тестов: {passed_tests}/{total_tests}")
    print(f"Процент успешности: {(passed_tests / total_tests) * 100:.1f}%")

    if passed_tests == total_tests:
        status = "🎉 ОТЛИЧНО"
        message = "Trust-ADE Protocol готов к использованию!"
    elif passed_tests >= total_tests * 0.75:
        status = "✅ ХОРОШО"
        message = "Trust-ADE Protocol в основном готов, есть небольшие проблемы"
    elif passed_tests >= total_tests * 0.5:
        status = "⚠️ ЧАСТИЧНО"
        message = "Trust-ADE Protocol частично функционален, требует доработки"
    else:
        status = "❌ ПРОБЛЕМЫ"
        message = "Trust-ADE Protocol требует серьезной отладки"

    print(f"\n{status}: {message}")

    return passed_tests == total_tests


if __name__ == "__main__":
    print("🚀 Запуск тестирования установки Trust-ADE Protocol")
    success = generate_installation_report()

    if success:
        print(f"\n🎯 Следующий шаг: запустите 'python examples/demo_trust_ade.py'")
    else:
        print(f"\n🔧 Рекомендуется устранить проблемы перед использованием")
