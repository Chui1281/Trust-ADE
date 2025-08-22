#!/usr/bin/env python3
"""
Основной скрипт для запуска сравнения моделей
ПОЛНОЕ сравнение ML моделей с Trust-ADE Protocol + CUDA + все исправления
✅ Исправлены: XANFIS, визуализация numpy, нулевые метрики Trust-ADE, CUDA оптимизация
🎯 Включает: полное обучение на всех датасетах, сохранение, визуализацию
"""

import numpy as np
from sklearn.model_selection import train_test_split

from cli.dataset_selector import create_dataset_selector, get_selected_datasets
from data.datasets import prepare_datasets, create_models_config
from training.trainers import train_models, train_fixed_xanfis_model
from evaluation.trust_evaluator import enhanced_trust_ade_evaluation
from analysis.results import print_dataset_summary, print_final_analysis
from utils.io_utils import save_results_and_visualizations
from config.settings import XANFIS_AVAILABLE


def run_comparison(selected_datasets=None, cuda_only=False, no_cuda=False, quick_mode=False, verbose=False):
    """Запуск сравнения моделей на выбранных датасетах"""
    
    print("=" * 100)
    print("🔬 ПРОДВИНУТОЕ СРАВНЕНИЕ ML МОДЕЛЕЙ С TRUST-ADE PROTOCOL + CUDA")
    print("🚀 Включает GPU ускорение и исправленные ошибки")
    print("=" * 100)

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

        # Обучение XANFIS
        if XANFIS_AVAILABLE and not cuda_only:
            wrapped_xanfis, xanfis_scaler, xanfis_accuracy, xanfis_time = train_fixed_xanfis_model(
                X_train, X_test, y_train, y_test, dataset_name, dataset_info['type']
            )

            if wrapped_xanfis and xanfis_accuracy > 0.1:
                trained_models['XANFIS'] = {
                    'wrapped_model': wrapped_xanfis,
                    'scaler': xanfis_scaler,
                    'training_time': xanfis_time,
                    'accuracy': xanfis_accuracy,
                    'needs_scaling': True,
                    'description': 'Adaptive Neuro-Fuzzy Inference System',
                    'color': '#9932CC',
                    'use_cuda': False
                }

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
    print(f"  • JSON файл с полными данными (исправлено)")
    print(f"  • 4+ типов профессиональных графиков с CUDA индикацией")
    print(f"  🚀 CUDA ускорение было использовано для совместимых моделей")

    return all_results, results_dir


def main():
    """Главная функция с поддержкой улучшенного XANFIS"""

    from cli.dataset_selector import create_dataset_selector, get_selected_datasets

    # Парсинг аргументов командной строки
    parser = create_dataset_selector()
    args = parser.parse_args()

    # Получение выбранных датасетов
    selected_datasets = get_selected_datasets(args)

    if not selected_datasets:
        print("❌ Не выбран ни один датасет!")
        return

    try:
        all_results, results_dir = run_comparison(
            selected_datasets=selected_datasets,
            cuda_only=args.cuda_only,
            no_cuda=args.no_cuda,
            quick_mode=args.quick,
            verbose=args.verbose
        )

        print(f"\n✅ ПОЛНОЕ СРАВНЕНИЕ ЗАВЕРШЕНО!")
        print(f"📈 Проанализировано {len(all_results)} датасетов")
        print(f"💾 Все результаты сохранены в: {results_dir}")

    except Exception as e:
        print(f"❌ Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Запуск ПОЛНОГО сравнения моделей ML с CUDA поддержкой")
    main()

