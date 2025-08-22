"""
CLI для выбора датасетов
"""
import argparse

def create_dataset_selector():
    """Создание CLI парсера для выбора датасетов"""
    
    parser = argparse.ArgumentParser(
        description='🔬 Trust-ADE ML Models Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Запуск на всех датасетах
  python main.py

  # Только конкретные датасеты
  python main.py --datasets iris breast_cancer

  # Все кроме больших датасетов
  python main.py --exclude digits_binary

  # Быстрое тестирование
  python main.py --datasets iris wine --quick

  # Только CUDA модели
  python main.py --cuda-only

Доступные датасеты:
  iris           : Классификация ирисов (3 класса, 4 признака)
  breast_cancer  : Диагностика рака молочной железы (2 класса, 30 признаков)
  wine          : Классификация вин (3 класса, 13 признаков)
  digits_binary : Распознавание цифры 0 (2 класса, 64 пикселя)
        """
    )
    
    # Выбор датасетов
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        choices=['iris', 'breast_cancer', 'wine', 'digits_binary'],
        help='Выбор конкретных датасетов'
    )
    
    parser.add_argument(
        '--exclude', '-e',
        nargs='+',
        choices=['iris', 'breast_cancer', 'wine', 'digits_binary'],
        help='Исключить датасеты'
    )
    
    # Дополнительные опции
    parser.add_argument(
        '--cuda-only',
        action='store_true',
        help='Использовать только CUDA модели'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Отключить CUDA модели'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Быстрый режим (меньше эпох, упрощенная оценка)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод'
    )
    
    return parser

def get_selected_datasets(args):
    """Получение списка выбранных датасетов"""
    
    all_datasets = ['iris', 'breast_cancer', 'wine', 'digits_binary']
    
    # Определяем базовый набор датасетов
    if args.datasets:
        selected = args.datasets
    else:
        selected = all_datasets.copy()
    
    # Исключаем указанные датасеты
    if args.exclude:
        selected = [d for d in selected if d not in args.exclude]
    
    return selected

