"""
Анализ и вывод результатов
"""
import numpy as np


def print_dataset_summary(dataset_name, trained_models):
    """Вывод результатов для датасета"""

    print(f"\n📊 РЕЗУЛЬТАТЫ ДЛЯ {dataset_name.upper()}:")

    # Заголовок таблицы
    print("Модель                              Точность   Trust Score  Уровень доверия      CUDA")
    print("-" * 90)

    # Сортировка по Trust Score
    sorted_models = sorted(
        trained_models.items(),
        key=lambda x: x[1].get('trust_results', {}).get('trust_score', 0),
        reverse=True
    )

    for model_name, model_info in sorted_models:
        accuracy = model_info.get('accuracy', 0.0)
        trust_results = model_info.get('trust_results', {})
        trust_score = trust_results.get('trust_score', 0.0)
        trust_level = trust_results.get('trust_level', 'Неизвестно')
        use_cuda = model_info.get('use_cuda', False)
        cuda_symbol = "🚀" if use_cuda else "💻"

        print(f"{model_name:<35} {accuracy:.3f}      {trust_score:.3f}        {trust_level:<20} {cuda_symbol}")


def print_final_analysis(all_results):
    """Финальный анализ всех результатов"""

    print("\n" + "=" * 100)
    print("🏆 ИТОГОВЫЙ АНАЛИЗ ВСЕХ ДАТАСЕТОВ (с CUDA поддержкой)")
    print("=" * 100)

    # Собираем статистику по моделям
    model_stats = {}

    for dataset_name, dataset_results in all_results.items():
        for model_name, model_info in dataset_results['models'].items():
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'trust_scores': [],
                    'accuracies': [],
                    'training_times': [],
                    'use_cuda': model_info.get('use_cuda', False)
                }

            trust_score = model_info.get('trust_results', {}).get('trust_score', 0.0)
            accuracy = model_info.get('accuracy', 0.0)
            training_time = model_info.get('training_time', 0.0)

            model_stats[model_name]['trust_scores'].append(trust_score)
            model_stats[model_name]['accuracies'].append(accuracy)
            model_stats[model_name]['training_times'].append(training_time)

    # Рейтинг по среднему Trust Score
    print(f"\n🎯 ОБЩИЙ РЕЙТИНГ МОДЕЛЕЙ (средний Trust Score):")

    model_rankings = []
    for model_name, stats in model_stats.items():
        if stats['trust_scores']:
            avg_trust = np.mean(stats['trust_scores'])
            std_trust = np.std(stats['trust_scores'])
            cuda_symbol = " 🚀" if stats['use_cuda'] else " 💻"
            model_rankings.append((model_name, avg_trust, std_trust, cuda_symbol))

    model_rankings.sort(key=lambda x: x[1], reverse=True)

    for i, (model_name, avg_trust, std_trust, cuda_symbol) in enumerate(model_rankings):
        rank_symbol = ["🥇", "🥈", "🥉"] + [f"{j}️⃣" for j in range(4, 10)]
        rank = rank_symbol[i] if i < len(rank_symbol) else f"{i + 1}️⃣"
        dataset_count = len(model_stats[model_name]['trust_scores'])
        print(f"  {rank} {model_name}: {avg_trust:.3f} ± {std_trust:.3f} (на {dataset_count} датасетах){cuda_symbol}")

