"""
Ввод/вывод и сохранение результатов
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from config.settings import project_root


def convert_numpy_types(obj):
    """Рекурсивно конвертирует numpy типы в стандартные Python типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def save_results_and_visualizations(all_results):
    """Сохранение результатов и создание визуализации с исправленной JSON сериализацией"""

    print("\n💾 Сохранение результатов и создание визуализации...")

    # Создание папки для результатов
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"  📁 Папка результатов: {results_dir}")

    # Временная метка
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Подготовка данных для сохранения
    detailed_results = []

    for dataset_name, dataset_results in all_results.items():
        for model_name, model_info in dataset_results['models'].items():
            # Детальные результаты
            trust_results = model_info.get('trust_results', {})
            detailed_row = {
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': float(model_info.get('accuracy', 0.0)),
                'Training_Time': float(model_info.get('training_time', 0.0)),
                'Trust_Score': float(trust_results.get('trust_score', 0.0)),
                'Trust_Level': trust_results.get('trust_level', 'Unknown'),
                'Explainability': float(trust_results.get('explainability_score', 0.0)),
                'Robustness': float(trust_results.get('robustness_index', 0.0)),
                'Bias_Shift': float(trust_results.get('bias_shift_index', 0.0)),
                'Concept_Drift': float(trust_results.get('concept_drift_rate', 0.0)),
                'CUDA': bool(model_info.get('use_cuda', False)),
                'Color': model_info.get('color', '#808080'),
                'Description': model_info.get('description', 'Unknown')
            }
            detailed_results.append(detailed_row)

    # Создание DataFrame
    df_detailed = pd.DataFrame(detailed_results)

    # Сохранение CSV файлов
    detailed_path = f'{results_dir}/detailed_comparison_cuda_{timestamp}.csv'
    df_detailed.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ Подробные результаты: {os.path.basename(detailed_path)}")

    # Создание визуализации
    print(f"  🎨 Создание визуализации...")
    try:
        from visualization.charts import create_fixed_visualizations
        create_fixed_visualizations(df_detailed, results_dir, timestamp)
        print(f"    ✅ Создано 4+ типов профессиональных графиков с CUDA индикацией")
    except Exception as viz_error:
        print(f"    ❌ Ошибка создания визуализации: {str(viz_error)}")

    print(f"  ✅ Все файлы сохранены в: {results_dir}")
    return results_dir

