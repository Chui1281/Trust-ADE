"""
Подготовка и загрузка датасетов
"""
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits


def prepare_datasets():
    """Подготовка всех датасетов с расширенной информацией"""

    datasets = {}

    print("📊 Подготовка датасетов...")

    # 1. Iris Dataset
    print("  🌸 Загружаем Iris Dataset...")
    iris = load_iris()
    datasets['iris'] = {
        'X': iris.data,
        'y': iris.target,
        'feature_names': list(iris.feature_names),
        'target_names': list(iris.target_names),
        'description': 'Классификация ирисов (3 класса, 4 признака)',
        'domain': 'general',
        'type': 'multiclass'
    }

    # 2. Breast Cancer Dataset
    print("  🏥 Загружаем Breast Cancer Dataset...")
    cancer = load_breast_cancer()
    datasets['breast_cancer'] = {
        'X': cancer.data,
        'y': cancer.target,
        'feature_names': list(cancer.feature_names),
        'target_names': list(cancer.target_names),
        'description': 'Диагностика рака молочной железы (2 класса, 30 признаков)',
        'domain': 'medical',
        'type': 'binary'
    }

    # 3. Wine Dataset
    print("  🍷 Загружаем Wine Dataset...")
    wine = load_wine()
    datasets['wine'] = {
        'X': wine.data,
        'y': wine.target,
        'feature_names': list(wine.feature_names),
        'target_names': list(wine.target_names),
        'description': 'Классификация вин (3 класса, 13 признаков)',
        'domain': 'general',
        'type': 'multiclass'
    }

    # 4. Digits Dataset (расширенный для лучших метрик Trust-ADE)
    print("  🔢 Загружаем Digits Dataset (расширенная версия)...")
    digits = load_digits()
    # Берем больше данных для лучших метрик Trust-ADE
    X_digits = digits.data[:1500]
    y_digits = (digits.target[:1500] == 0).astype(int)

    datasets['digits_binary'] = {
        'X': X_digits,
        'y': y_digits,
        'feature_names': [f"pixel_{i}" for i in range(X_digits.shape[1])],
        'target_names': ['not_zero', 'zero'],
        'description': 'Распознавание цифры 0 (2 класса, 64 пикселя)',
        'domain': 'general',
        'type': 'binary'
    }

    print(f"✅ Подготовлено {len(datasets)} датасетов")
    return datasets


def create_models_config():
    """Создание конфигурации моделей с улучшенными параметрами"""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    models_config = {
        'Random Forest': {
            'sklearn_class': RandomForestClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 3,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'random_state': 42,
                'n_jobs': -1
            },
            'needs_scaling': False,
            'description': 'Ансамбль решающих деревьев',
            'color': '#2E8B57',
            'use_cuda': False
        },

        'MLP Neural Network (CPU)': {
            'sklearn_class': MLPClassifier,
            'params': {
                'hidden_layer_sizes': (150, 100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'random_state': 42,
                'early_stopping': True,
                'validation_fraction': 0.15
            },
            'needs_scaling': True,
            'description': 'Многослойный персептрон (CPU)',
            'color': '#4169E1',
            'use_cuda': False
        },

        'Support Vector Machine': {
            'sklearn_class': SVC,
            'params': {
                'kernel': 'rbf',
                'C': 2.0,
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            },
            'needs_scaling': True,
            'description': 'Метод опорных векторов',
            'color': '#DC143C',
            'use_cuda': False
        },

        'Gradient Boosting': {
            'sklearn_class': GradientBoostingClassifier,
            'params': {
                'n_estimators': 150,
                'learning_rate': 0.15,
                'max_depth': 4,
                'min_samples_split': 3,
                'subsample': 0.9,
                'random_state': 42
            },
            'needs_scaling': False,
            'description': 'Градиентный бустинг',
            'color': '#FF8C00',
            'use_cuda': False
        }
    }

    return models_config

