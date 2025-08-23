"""
Тренировка моделей с улучшенным XANFIS
"""
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config.settings import CUDA_AVAILABLE, DEVICE, XANFIS_AVAILABLE
from models.wrappers import SklearnWrapper, CUDAMLPWrapper, FixedXANFISWrapper
from models.cuda_models import OptimizedCUDAMLPClassifier

# Импорт улучшенного XANFIS из папки models
if XANFIS_AVAILABLE:
    try:
        from models.xanfis_wrapper import (
            TrustAdeCompatibleXANFIS,
            TrustAdeXANFISWrapper,
            train_improved_xanfis_model
        )
        ENHANCED_XANFIS_AVAILABLE = True
        print("✅ Улучшенный XANFIS загружен из models/")
    except ImportError as e:
        print(f"⚠️ Не удалось загрузить улучшенный XANFIS: {e}")
        ENHANCED_XANFIS_AVAILABLE = False
else:
    ENHANCED_XANFIS_AVAILABLE = False


def train_fixed_xanfis_model(X_train, X_test, y_train, y_test, dataset_name, dataset_type):
    """Обучение XANFIS с использованием улучшенного wrapper из models/"""

    if not XANFIS_AVAILABLE:
        return None, None, 0.0, 0.0

    # Используем улучшенную версию если доступна
    if ENHANCED_XANFIS_AVAILABLE:
        print(f"    🧠 Обучение улучшенного XANFIS на {dataset_name}...")

        # Подготавливаем имена признаков
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Вызываем улучшенную функцию обучения
        wrapped_xanfis, scaler, accuracy, training_time = train_improved_xanfis_model(
            X_train, X_test, y_train, y_test, dataset_name, feature_names,dataset_type
        )

        if wrapped_xanfis and accuracy > 0.1:
            print(f"      ✅ Улучшенный XANFIS обучен успешно!")
            print(f"      📊 Точность: {accuracy:.3f}")
            print(f"      ⏱️ Время: {training_time:.2f}s")
            print(f"      🧠 Правил: {len(wrapped_xanfis.get_fuzzy_rules())}")
            return wrapped_xanfis, scaler, accuracy, training_time

def train_models(X_train, X_test, y_train, y_test, feature_names, models_config, dataset_type, dataset_name):
    """Обучение всех моделей включая улучшенный XANFIS и CUDA"""

    trained_models = {}
    n_samples = len(X_train)

    # Обучение стандартных sklearn моделей
    for model_name, config in models_config.items():
        print(f"  📈 Обучение {model_name}...")

        try:
            start_time = time.time()

            # Обычные sklearn модели
            model = config['sklearn_class'](**config['params'])

            # Подготовка данных
            if config['needs_scaling']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                scaler = None
                X_train_scaled, X_test_scaled = X_train, X_test

            # Обучение
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            # Оценка
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Обертка
            wrapped_model = SklearnWrapper(model, feature_names)

            print(f"    ✅ {model_name} обучен за {training_time:.2f} сек, точность: {accuracy:.3f}")

            trained_models[model_name] = {
                'wrapped_model': wrapped_model,
                'scaler': scaler,
                'training_time': training_time,
                'accuracy': accuracy,
                'needs_scaling': config['needs_scaling'],
                'description': config['description'],
                'color': config['color'],
                'use_cuda': config.get('use_cuda', False),
                'model_type': 'sklearn'
            }

        except Exception as e:
            print(f"    ❌ Ошибка обучения {model_name}: {str(e)}")
            continue

    # CUDA модель
    if CUDA_AVAILABLE:
        cuda_model_name = 'MLP Neural Network (CUDA)'
        print(f"  📈 Обучение {cuda_model_name}...")

        try:
            start_time = time.time()

            # Определяем количество классов для CUDA MLP
            n_classes = len(np.unique(y_train))

            model = OptimizedCUDAMLPClassifier(
                hidden_layers=(128, 64),
                n_classes=n_classes,
                learning_rate=0.001,
                epochs=200,
                device=DEVICE,
                random_state=42,
                dataset_size=n_samples
            )

            # Обучение
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Оценка
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Обертка
            wrapped_model = CUDAMLPWrapper(model, feature_names)

            cuda_used = "🚀" if model.use_cuda else "📱"
            print(f"    ✅ {cuda_model_name} обучен за {training_time:.2f} сек, точность: {accuracy:.3f}")
            print(f"    {cuda_used} Использовалось {'CUDA' if model.use_cuda else 'CPU'} ускорение")

            trained_models[cuda_model_name] = {
                'wrapped_model': wrapped_model,
                'scaler': None,
                'training_time': training_time,
                'accuracy': accuracy,
                'needs_scaling': False,
                'description': f'Многослойный персептрон ({"CUDA" if model.use_cuda else "CPU адаптивный"})',
                'color': '#8A2BE2',
                'use_cuda': model.use_cuda,
                'model_type': 'cuda'
            }

        except Exception as e:
            print(f"    ❌ Ошибка обучения {cuda_model_name}: {str(e)}")

    return trained_models


def run_comparison(selected_datasets=None, cuda_only=False, no_cuda=False, quick_mode=False, verbose=False):
    """Запуск сравнения моделей на выбранных датасетах с улучшенным XANFIS"""

    print("=" * 100)
    print("🔬 ПРОДВИНУТОЕ СРАВНЕНИЕ ML МОДЕЛЕЙ С TRUST-ADE + УЛУЧШЕННЫЙ XANFIS")
    print("🚀 Включает GPU ускорение и исправленный XANFIS из models/")
    print("=" * 100)

    # Импорт необходимых функций
    from cli.dataset_selector import get_selected_datasets
    from data.datasets import prepare_datasets, create_models_config
    from evaluation.trust_evaluator import enhanced_trust_ade_evaluation
    from analysis.results import print_dataset_summary, print_final_analysis
    from utils.io_utils import save_results_and_visualizations

    # Подготовка данных
    all_datasets = prepare_datasets()
    models_config = create_models_config()

    # Фильтруем датасеты если указаны конкретные
    if selected_datasets:
        datasets = {name: info for name, info in all_datasets.items()
                   if name in selected_datasets}
        print(f"📊 Выбранные датасеты: {', '.join(selected_datasets)}")
    else:
        datasets = all_datasets
        print(f"📊 Используем все датасеты: {', '.join(datasets.keys())}")

    # Статус XANFIS
    if ENHANCED_XANFIS_AVAILABLE:
        print("✅ Улучшенный XANFIS активен из models/improved_xanfis.py")
        print("   🧠 Объяснимость: Полная поддержка правил")
        print("   🔧 Исправления: Устранены ошибки форматирования")
    elif XANFIS_AVAILABLE:
        print("⚠️ Доступен только базовый XANFIS (fallback)")
    else:
        print("❌ XANFIS полностью недоступен")

    all_results = {}

    for dataset_name, dataset_info in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"📊 ТЕСТИРОВАНИЕ НА ДАТАСЕТЕ: {dataset_name.upper()}")
        print(f"📋 Описание: {dataset_info['description']}")
        print(f"🏷️ Домен: {dataset_info['domain']}")
        print(f"🎯 Тип задачи: {dataset_info['type']}")
        print("=" * 80)

        # Подготовка данных
        X, y = dataset_info['X'], dataset_info['y']
        print(f"📊 Размер данных: {X.shape[0]} образцов, {X.shape[1]} признаков")
        print(f"📊 Классы: {np.bincount(y)} (общее количество: {len(np.unique(y))})")

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"📊 Разделение: {len(X_train)} обучение / {len(X_test)} тест")

        # Обучение моделей
        print(f"\n🤖 Обучение моделей...")
        trained_models = train_models(
            X_train, X_test, y_train, y_test,
            dataset_info['feature_names'], models_config,
            dataset_info['type'], dataset_name
        )

        # Обучение XANFIS с улучшенной версией
        if XANFIS_AVAILABLE and not cuda_only:
            wrapped_xanfis, xanfis_scaler, xanfis_accuracy, xanfis_time = train_fixed_xanfis_model(
                X_train, X_test, y_train, y_test, dataset_name, dataset_info['type']
            )

            if wrapped_xanfis and xanfis_accuracy > 0.1:
                # Определяем тип XANFIS для описания
                xanfis_description = ('Trust-ADE Compatible XANFIS (улучшенная объяснимость)'
                                    if ENHANCED_XANFIS_AVAILABLE
                                    else 'Adaptive Neuro-Fuzzy Inference System (базовая версия)')

                trained_models['XANFIS'] = {
                    'wrapped_model': wrapped_xanfis,
                    'scaler': xanfis_scaler,
                    'training_time': xanfis_time,
                    'accuracy': xanfis_accuracy,
                    'needs_scaling': True,
                    'description': xanfis_description,
                    'color': '#9932CC',
                    'use_cuda': False,
                    'model_type': 'enhanced_xanfis' if ENHANCED_XANFIS_AVAILABLE else 'legacy_xanfis',
                    'explainability': 'high' if ENHANCED_XANFIS_AVAILABLE else 'medium'
                }

                # Дополнительная информация для улучшенного XANFIS
                if ENHANCED_XANFIS_AVAILABLE and hasattr(wrapped_xanfis, 'get_fuzzy_rules'):
                    try:
                        rules_count = len(wrapped_xanfis.get_fuzzy_rules())
                        trained_models['XANFIS']['rules_count'] = rules_count
                        print(f"    📋 Извлечено правил: {rules_count}")
                    except Exception:
                        pass

        # Trust-ADE оценка
        enhanced_trust_ade_evaluation(trained_models, X_test, y_test, dataset_info['domain'], X_train)

        # Сохранение результатов
        all_results[dataset_name] = {
            'dataset_info': dataset_info,
            'models': trained_models,
            'X_test_shape': X_test.shape,
            'y_test_shape': y_test.shape
        }

        # Вывод результатов
        print_dataset_summary(dataset_name, trained_models)

    # Финальный анализ и сохранение
    print_final_analysis(all_results)
    results_dir = save_results_and_visualizations(all_results)

    print(f"\n🎉 СРАВНЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print(f"📁 Результаты и графики сохранены в: {results_dir}")
    print(f"📊 Создано:")
    print(f"  • CSV файлы с подробными и краткими результатами")
    print(f"  • JSON файл с полными данными")
    print(f"  • 4+ типов профессиональных графиков с CUDA и XANFIS индикацией")

    if ENHANCED_XANFIS_AVAILABLE:
        print(f"  🧠 Улучшенный XANFIS с исправленной объяснимостью")

    print(f"  🚀 CUDA ускорение было использовано для совместимых моделей")

    return all_results, results_dir

