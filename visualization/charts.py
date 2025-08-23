"""
Расширенный модуль визуализации для Trust-ADE Protocol
Включает основные и дополнительные типы графиков для анализа доверия к ML моделям
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import warnings
from math import pi
import os

warnings.filterwarnings('ignore')

# Настройка стиля matplotlib
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Цветовая палитра для разных типов моделей
MODEL_COLORS = {
    'Support Vector Machine': '#DC143C',
    'Random Forest': '#2E8B57', 
    'Gradient Boosting': '#FF8C00',
    'MLP Neural Network (CPU)': '#4169E1',
    'MLP Neural Network (CUDA)': '#8A2BE2',
    'XANFIS': '#9932CC',
    'Default': '#808080'
}

def create_extended_visualizations(df_viz, results_dir, timestamp):
    """Создание расширенного набора визуализаций"""
    
    print(f"  🎨 Создание расширенной визуализации Trust-ADE...")
    
    try:
        # Создаем директорию если не существует
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Основные графики (исправленные версии)
        create_fixed_main_comparison(df_viz, results_dir, timestamp)
        create_trust_metrics_analysis(df_viz, results_dir, timestamp)
        
        # 2. CUDA vs CPU анализ (если есть CUDA данные)
        if 'CUDA' in df_viz.columns and any(df_viz['CUDA']):
            create_cuda_performance_comparison(df_viz, results_dir, timestamp)
        
        # 3. Корреляционный анализ
        create_correlation_analysis(df_viz, results_dir, timestamp)
        
        # 4. НОВЫЕ ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ
        create_radar_chart_comparison(df_viz, results_dir, timestamp)
        create_performance_matrix(df_viz, results_dir, timestamp)
        create_dataset_comparison(df_viz, results_dir, timestamp)
        create_metric_distributions(df_viz, results_dir, timestamp)
        create_trust_score_breakdown(df_viz, results_dir, timestamp)
        create_efficiency_analysis(df_viz, results_dir, timestamp)
        create_model_ranking_chart(df_viz, results_dir, timestamp)
        create_detailed_heatmap(df_viz, results_dir, timestamp)
        
        print(f"    ✅ Создано 12 типов профессиональных графиков")
        
    except Exception as e:
        print(f"    ❌ Ошибка создания визуализации: {str(e)}")
        import traceback
        traceback.print_exc()


def create_fixed_main_comparison(df_viz, results_dir, timestamp):
    """Основной график сравнения моделей с исправленным форматированием"""
    
    try:
        plt.figure(figsize=(16, 10))
        
        # Группируем по моделям
        model_stats = df_viz.groupby('Model').agg({
            'Accuracy': 'mean',
            'Trust_Score': 'mean',
            'CUDA': 'first',
            'Color': 'first'
        }).reset_index()
        
        models = model_stats['Model'].values
        accuracy_means = model_stats['Accuracy'].values.astype(float)
        trust_means = model_stats['Trust_Score'].values.astype(float)
        colors = [MODEL_COLORS.get(model, MODEL_COLORS['Default']) for model in models]
        cuda_flags = model_stats['CUDA'].values if 'CUDA' in model_stats.columns else [False] * len(models)
        
        x = np.arange(len(models))
        width = 0.35
        
        # Создаем столбцы
        bars1 = plt.bar(x - width / 2, accuracy_means, width, label='Accuracy',
                        color='lightblue', alpha=0.8, edgecolor='navy')
        bars2 = plt.bar(x + width / 2, trust_means, width, label='Trust Score',
                        color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Добавляем значения на столбцы
        for i, bar in enumerate(bars1):
            height = float(bar.get_height())
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for i, bar in enumerate(bars2):
            height = float(bar.get_height())
            cuda_symbol = " 🚀" if cuda_flags[i] else ""
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}{cuda_symbol}', ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('🏆 Trust-ADE: Model Accuracy and Trust Score Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/main_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_fixed_main_comparison: {e}")
        plt.close()


def create_trust_metrics_analysis(df_viz, results_dir, timestamp):
    """Детальный анализ всех Trust-ADE метрик"""
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔍 Detailed Trust-ADE Metrics Analysis', fontsize=16, fontweight='bold')
        metrics = [
            ('Trust_Score', 'Trust Score', 'viridis'),
            ('Explainability', 'Explainability', 'Blues'),
            ('Robustness', 'Robustness', 'Greens'),
            ('Bias_Shift', 'Bias Shift', 'Reds'),
            ('Concept_Drift', 'Concept Drift', 'Purples'),
            ('Training_Time', 'Training Time (s)', 'Oranges')
        ]

        for idx, (metric, title, colormap) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            if metric in df_viz.columns:
                model_means = df_viz.groupby('Model')[metric].mean().sort_values(ascending=False)
                values = [float(x) for x in model_means.values]
                
                if values:
                    bars = ax.bar(range(len(model_means)), values,
                                  color=plt.cm.get_cmap(colormap)(0.7), alpha=0.8,
                                  edgecolor='black', linewidth=1)
                    
                    for i, bar in enumerate(bars):
                        height = float(bar.get_height())
                        format_str = f'{height:.3f}' if metric != 'Training_Time' else f'{height:.2f}s'
                        ax.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                                format_str, ha='center', va='bottom', fontweight='bold', fontsize=8)
                    
                    ax.set_title(f'📈 {title}', fontweight='bold')
                    ax.set_xticks(range(len(model_means)))
                    ax.set_xticklabels(model_means.index, rotation=45, ha='right')
                    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
                    ax.grid(axis='y', alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Нет данных\nдля метрики\n{metric}',
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, f'Метрика\n{metric}\nнедоступна',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'❌ {title}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/trust_metrics_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_trust_metrics_analysis: {e}")
        plt.close()


def create_cuda_performance_comparison(df_viz, results_dir, timestamp):
    """Сравнение CUDA vs CPU производительности"""
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚀 CUDA vs CPU: Performance Comparison', fontsize=16, fontweight='bold')
        
        cuda_data = df_viz[df_viz['CUDA'] == True]
        cpu_data = df_viz[df_viz['CUDA'] == False]
        
        if len(cuda_data) > 0 and len(cpu_data) > 0:
            categories = ['CUDA Models', 'CPU Models']
            
            # График 1: Trust Score
            trust_means = [float(cuda_data['Trust_Score'].mean()), float(cpu_data['Trust_Score'].mean())]
            trust_stds = [float(cuda_data['Trust_Score'].std()), float(cpu_data['Trust_Score'].std())]
            trust_stds = [std if not np.isnan(std) else 0 for std in trust_stds]
            
            bars1 = ax1.bar(categories, trust_means, yerr=trust_stds,
                            color=['#FFD700', '#C0C0C0'], alpha=0.8,
                            edgecolor='black', capsize=5)
            
            for bar, mean in zip(bars1, trust_means):
                ax1.text(bar.get_x() + bar.get_width() / 2., mean + 0.01,
                         f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_title('🎯 Trust Score Comparison')
            ax1.set_ylabel('Average Trust Score')
            ax1.grid(axis='y', alpha=0.3)
            
            # График 2: Время обучения
            time_means = [float(cuda_data['Training_Time'].mean()), float(cpu_data['Training_Time'].mean())]
            time_stds = [float(cuda_data['Training_Time'].std()), float(cpu_data['Training_Time'].std())]
            time_stds = [std if not np.isnan(std) else 0 for std in time_stds]
            
            bars2 = ax2.bar(categories, time_means, yerr=time_stds,
                            color=['#FFD700', '#C0C0C0'], alpha=0.8,
                            edgecolor='black', capsize=5)
            
            for bar, mean in zip(bars2, time_means):
                ax2.text(bar.get_x() + bar.get_width() / 2., mean * 1.1,
                         f'{mean:.2f}s', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_title('⚡ Training Time Comparison')
            ax2.set_ylabel('Average Training Time (seconds)')
            ax2.grid(axis='y', alpha=0.3)
            
            # График 3: Точность
            acc_means = [float(cuda_data['Accuracy'].mean()), float(cpu_data['Accuracy'].mean())]
            acc_stds = [float(cuda_data['Accuracy'].std()), float(cpu_data['Accuracy'].std())]
            acc_stds = [std if not np.isnan(std) else 0 for std in acc_stds]
            
            bars3 = ax3.bar(categories, acc_means, yerr=acc_stds,
                            color=['#FFD700', '#C0C0C0'], alpha=0.8,
                            edgecolor='black', capsize=5)
            
            for bar, mean in zip(bars3, acc_means):
                ax3.text(bar.get_x() + bar.get_width() / 2., mean + 0.01,
                         f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax3.set_title('📊 Accuracy Comparison')
            ax3.set_ylabel('Average Accuracy')
            ax3.grid(axis='y', alpha=0.3)
            
            # График 4: Эффективность
            cuda_eff = float(cuda_data['Trust_Score'].mean() / max(cuda_data['Training_Time'].mean(), 0.001))
            cpu_eff = float(cpu_data['Trust_Score'].mean() / max(cpu_data['Training_Time'].mean(), 0.001))
            
            bars4 = ax4.bar(categories, [cuda_eff, cpu_eff],
                            color=['#FFD700', '#C0C0C0'], alpha=0.8, edgecolor='black')
            
            for bar, eff in zip(bars4, [cuda_eff, cpu_eff]):
                ax4.text(bar.get_x() + bar.get_width() / 2., eff * 1.05,
                         f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
            
            ax4.set_title('⚖️ Efficiency (Trust Score / Time)')
            ax4.set_ylabel('Efficiency Ratio')
            ax4.grid(axis='y', alpha=0.3)
        
        else:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'Недостаточно данных\nдля CUDA vs CPU\nсравнения',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/cuda_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_cuda_performance_comparison: {e}")
        plt.close()


def create_correlation_analysis(df_viz, results_dir, timestamp):
    """Корреляционный анализ между метриками"""
    
    try:
        plt.figure(figsize=(12, 10))
        
        numeric_columns = ['Accuracy', 'Trust_Score', 'Explainability', 'Robustness',
                           'Bias_Shift', 'Concept_Drift', 'Training_Time']
        
        available_columns = [col for col in numeric_columns if col in df_viz.columns]
        
        if len(available_columns) > 1:
            corr_data = df_viz[available_columns].astype(float)
            correlation_matrix = corr_data.corr()
            
            if not correlation_matrix.empty:
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                
                sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                            square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                            fmt='.3f', annot_kws={'fontweight': 'bold'})
                
                plt.title('🔗 Correlation Analysis of Trust-ADE Metrics',
                          fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Metrics', fontweight='bold')
                plt.ylabel('Metrics', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
            else:
                plt.text(0.5, 0.5, 'Нет данных для\nкорреляционного\nанализа',
                         ha='center', va='center', transform=plt.gca().transAxes,
                         fontsize=14, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'Недостаточно\nметрик для\nанализа корреляции',
                     ha='center', va='center', transform=plt.gca().transAxes,
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/correlation_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_correlation_analysis: {e}")
        plt.close()


def create_radar_chart_comparison(df_viz, results_dir, timestamp):
    """Радарная диаграмма для сравнения моделей по всем метрикам"""
    
    try:
        # Подготовка данных для радарной диаграммы
        radar_metrics = ['Trust_Score', 'Accuracy', 'Explainability', 'Robustness']
        available_metrics = [m for m in radar_metrics if m in df_viz.columns]
        
        if len(available_metrics) < 3:
            return
        
        model_stats = df_viz.groupby('Model')[available_metrics].mean()
        
        # Настройка радарной диаграммы
        angles = [n / float(len(available_metrics)) * 2 * pi for n in range(len(available_metrics))]
        angles += angles[:1]  # Замыкаем круг
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_stats)))
        
        for idx, (model, values) in enumerate(model_stats.iterrows()):
            values_list = values.tolist()
            values_list += values_list[:1]  # Замыкаем круг
            
            ax.plot(angles, values_list, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values_list, alpha=0.25, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('🕸️ Radar Comparison of Models by Trust-ADE Metrics',
                     size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/radar_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_radar_chart_comparison: {e}")
        plt.close()


def create_performance_matrix(df_viz, results_dir, timestamp):
    """Матрица производительности: Trust Score vs Accuracy vs Training Time"""
    
    try:
        plt.figure(figsize=(12, 8))
        
        # Создаем scatter plot
        scatter = plt.scatter(df_viz['Accuracy'], df_viz['Trust_Score'], 
                             s=df_viz['Training_Time'] * 1000,  # Размер точки = время обучения
                             c=[MODEL_COLORS.get(model, MODEL_COLORS['Default']) for model in df_viz['Model']],
                             alpha=0.7, edgecolors='black', linewidth=1)
        
        # Добавляем подписи моделей
        for idx, row in df_viz.iterrows():
            plt.annotate(row['Model'], (row['Accuracy'], row['Trust_Score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.xlabel('Accuracy', fontsize=12, fontweight='bold')
        plt.ylabel('Trust Score', fontsize=12, fontweight='bold')
        plt.title('🎯 Model Performance Matrix\n(point size = training time)',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Создаем легенду для размера точек
        sizes = [0.1, 0.3, 0.5]
        size_labels = ['Быстро', 'Средне', 'Медленно']
        size_legend = [plt.scatter([], [], s=s*1000, c='gray', alpha=0.7) for s in sizes]
        plt.legend(size_legend, size_labels, title='Время обучения', 
                  loc='upper left', title_fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/performance_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_performance_matrix: {e}")
        plt.close()


def create_dataset_comparison(df_viz, results_dir, timestamp):
    """Сравнение производительности моделей по датасетам"""
    
    try:
        if 'Dataset' not in df_viz.columns:
            return
        
        datasets = df_viz['Dataset'].unique()
        if len(datasets) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 Model Comparison by Dataset', fontsize=16, fontweight='bold')
        
        metrics = ['Trust_Score', 'Accuracy', 'Training_Time', 'Explainability']
        metric_titles = ['Trust Score', 'Accuracy', 'Training Time (s)', 'Explainability']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            if metric not in df_viz.columns:
                continue
                
            ax = axes[idx // 2, idx % 2]
            
            pivot_data = df_viz.pivot_table(values=metric, index='Model', columns='Dataset', aggfunc='mean')
            
            im = ax.imshow(pivot_data.values, aspect='auto', cmap='viridis', interpolation='nearest')
            
            # Настройки осей
            ax.set_xticks(range(len(pivot_data.columns)))
            ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels(pivot_data.index)
            
            # Добавляем значения в ячейки
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    value = pivot_data.iloc[i, j]
                    if not np.isnan(value):
                        ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                               color='white' if value < pivot_data.values.mean() else 'black',
                               fontweight='bold', fontsize=8)
            
            ax.set_title(f'📈 {title}', fontweight='bold')
            
            # Цветовая шкала
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/dataset_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_dataset_comparison: {e}")
        plt.close()


def create_metric_distributions(df_viz, results_dir, timestamp):
    """Распределения Trust-ADE метрик"""
    
    try:
        metrics = ['Trust_Score', 'Accuracy', 'Explainability', 'Robustness']
        available_metrics = [m for m in metrics if m in df_viz.columns]
        
        if len(available_metrics) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 Trust-ADE Metric Distributions', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(available_metrics[:4]):
            ax = axes[idx // 2, idx % 2]
            
            # Гистограмма с kde
            ax.hist(df_viz[metric], bins=20, alpha=0.7, edgecolor='black', 
                   color=plt.cm.viridis(0.7), density=True)
            
            # KDE кривая
            from scipy import stats
            kde = stats.gaussian_kde(df_viz[metric].dropna())
            x_range = np.linspace(df_viz[metric].min(), df_viz[metric].max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            # Статистики
            mean_val = df_viz[metric].mean()
            std_val = df_viz[metric].std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, 
                      label=f'Mean + Std: {mean_val + std_val:.3f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7,
                      label=f'Mean - Std: {mean_val - std_val:.3f}')
            
            ax.set_title(f'📈 Distribution {metric}', fontweight='bold')
            ax.set_xlabel(metric)
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/metric_distributions_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_metric_distributions: {e}")
        plt.close()


def create_trust_score_breakdown(df_viz, results_dir, timestamp):
    """Детальная разбивка Trust Score по компонентам"""
    
    try:
        trust_components = ['Explainability', 'Robustness']
        available_components = [c for c in trust_components if c in df_viz.columns]
        
        if len(available_components) < 2:
            return
        
        model_stats = df_viz.groupby('Model')[available_components + ['Trust_Score']].mean()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('🔍 Trust Score Detailed Breakdown', fontsize=16, fontweight='bold')
        
        # Stacked bar chart компонентов
        bottom = np.zeros(len(model_stats))
        colors = plt.cm.Set3(np.linspace(0, 1, len(available_components)))
        
        for idx, component in enumerate(available_components):
            ax1.bar(model_stats.index, model_stats[component], bottom=bottom, 
                   label=component, color=colors[idx], alpha=0.8)
            bottom += model_stats[component]
        
        ax1.set_title('📊 Trust Score Components', fontweight='bold')
        ax1.set_ylabel('Component Value')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot: компоненты vs итоговый Trust Score
        if len(available_components) >= 2:
            scatter = ax2.scatter(model_stats[available_components[0]], 
                                 model_stats[available_components[1]],
                                 s=model_stats['Trust_Score'] * 500,
                                 c=model_stats['Trust_Score'], 
                                 cmap='viridis', alpha=0.7, edgecolors='black')
            
            # Подписи моделей
            for idx, model in enumerate(model_stats.index):
                ax2.annotate(model, 
                           (model_stats.iloc[idx][available_components[0]], 
                            model_stats.iloc[idx][available_components[1]]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            ax2.set_xlabel(available_components[0])
            ax2.set_ylabel(available_components[1])
            ax2.set_title('🎯 Components vs Trust Score\n(point size = Trust Score)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Цветовая шкала
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Trust Score')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/trust_score_breakdown_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_trust_score_breakdown: {e}")
        plt.close()


def create_efficiency_analysis(df_viz, results_dir, timestamp):
    """Анализ эффективности моделей"""
    
    try:
        # Вычисляем различные метрики эффективности
        df_viz = df_viz.copy()
        df_viz['Trust_Efficiency'] = df_viz['Trust_Score'] / (df_viz['Training_Time'] + 0.001)
        df_viz['Accuracy_Efficiency'] = df_viz['Accuracy'] / (df_viz['Training_Time'] + 0.001)
        df_viz['Overall_Efficiency'] = (df_viz['Trust_Score'] + df_viz['Accuracy']) / 2 / (df_viz['Training_Time'] + 0.001)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('⚡ Model Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # График 1: Trust Efficiency
        model_eff = df_viz.groupby('Model')['Trust_Efficiency'].mean().sort_values(ascending=False)
        bars1 = ax1.bar(range(len(model_eff)), model_eff.values, 
                       color=plt.cm.viridis(0.7), alpha=0.8, edgecolor='black')
        ax1.set_title('🎯 Trust Efficiency (Trust Score / Time)', fontweight='bold')
        ax1.set_xticks(range(len(model_eff)))
        ax1.set_xticklabels(model_eff.index, rotation=45, ha='right')
        ax1.set_ylabel('Trust Score / Second')
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars1, model_eff.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_eff) * 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # График 2: Accuracy Efficiency  
        acc_eff = df_viz.groupby('Model')['Accuracy_Efficiency'].mean().sort_values(ascending=False)
        bars2 = ax2.bar(range(len(acc_eff)), acc_eff.values,
                       color=plt.cm.plasma(0.7), alpha=0.8, edgecolor='black')
        ax2.set_title('📊 Accuracy Efficiency (Accuracy / Time)', fontweight='bold')
        ax2.set_xticks(range(len(acc_eff)))
        ax2.set_xticklabels(acc_eff.index, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy / Second')
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, acc_eff.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(acc_eff) * 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # График 3: Overall Efficiency
        overall_eff = df_viz.groupby('Model')['Overall_Efficiency'].mean().sort_values(ascending=False)
        bars3 = ax3.bar(range(len(overall_eff)), overall_eff.values,
                       color=plt.cm.cividis(0.7), alpha=0.8, edgecolor='black')
        ax3.set_title('⚖️ Overall Efficiency', fontweight='bold')
        ax3.set_xticks(range(len(overall_eff)))
        ax3.set_xticklabels(overall_eff.index, rotation=45, ha='right')
        ax3.set_ylabel('Overall Score / Second')
        ax3.grid(True, alpha=0.3)
        
        for bar, val in zip(bars3, overall_eff.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(overall_eff) * 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # График 4: Scatter - Quality vs Speed
        quality_score = (df_viz['Trust_Score'] + df_viz['Accuracy']) / 2
        ax4.scatter(df_viz['Training_Time'], quality_score, 
                   s=100, alpha=0.7, edgecolors='black')
        
        for idx, row in df_viz.iterrows():
            ax4.annotate(row['Model'], (row['Training_Time'], quality_score.iloc[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('Quality Score (Trust + Accuracy)/2')
        ax4.set_title('🎯 Quality vs Speed Trade-off', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/efficiency_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_efficiency_analysis: {e}")
        plt.close()


def create_model_ranking_chart(df_viz, results_dir, timestamp):
    """Рейтинг моделей по различным критериям"""
    
    try:
        # Создаем рейтинги по разным метрикам
        metrics_for_ranking = ['Trust_Score', 'Accuracy', 'Explainability', 'Robustness']
        available_metrics = [m for m in metrics_for_ranking if m in df_viz.columns]
        
        if len(available_metrics) < 2:
            return
        
        model_stats = df_viz.groupby('Model')[available_metrics].mean()
        
        # Рассчитываем ранги (1 = лучший)
        rankings = model_stats.rank(ascending=False, method='min')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('🏆 Model Ranking by Trust-ADE protocol', fontsize=16, fontweight='bold')
        
        # Тепловая карта рангов
        im = ax1.imshow(rankings.T, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
        
        ax1.set_xticks(range(len(rankings.index)))
        ax1.set_xticklabels(rankings.index, rotation=45, ha='right')
        ax1.set_yticks(range(len(available_metrics)))
        ax1.set_yticklabels(available_metrics)
        
        # Добавляем ранги в ячейки
        for i in range(len(available_metrics)):
            for j in range(len(rankings.index)):
                rank = int(rankings.iloc[j, i])
                ax1.text(j, i, f'{rank}', ha='center', va='center',
                        color='white' if rank > len(rankings.index)/2 else 'black',
                        fontweight='bold', fontsize=12)
        
        ax1.set_title('📊 Rank Matrix\n(1 = best)', fontweight='bold')
        
        # Цветовая шкала
        cbar1 = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar1.set_label('Rank')
        
        # Общий рейтинг (средний ранг)
        overall_ranking = rankings.mean(axis=1).sort_values()
        
        colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Золото, серебро, бронза
        colors.extend(plt.cm.Set3(np.linspace(0, 1, max(0, len(overall_ranking) - 3))))
        
        bars = ax2.barh(range(len(overall_ranking)), overall_ranking.values,
                       color=colors[:len(overall_ranking)], alpha=0.8, edgecolor='black')
        
        ax2.set_yticks(range(len(overall_ranking)))
        ax2.set_yticklabels(overall_ranking.index)
        ax2.set_xlabel('Average Rank')
        ax2.set_title('🥇 Overall Model Ranking\n(lower = better)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Добавляем значения рангов
        for bar, rank in zip(bars, overall_ranking.values):
            ax2.text(rank + max(overall_ranking) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{rank:.1f}', ha='left', va='center', fontweight='bold')
        
        # Добавляем медали для топ-3
        medals = ['🥇', '🥈', '🥉']
        for i, (model, rank) in enumerate(overall_ranking.head(3).items()):
            ax2.text(0.02, i, medals[i], transform=ax2.transData, fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/model_ranking_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_model_ranking_chart: {e}")
        plt.close()


def create_detailed_heatmap(df_viz, results_dir, timestamp):
    """Детальная тепловая карта всех метрик по моделям"""
    
    try:
        # Выбираем все числовые метрики
        numeric_columns = ['Trust_Score', 'Accuracy', 'Explainability', 'Robustness',
                          'Bias_Shift', 'Concept_Drift', 'Training_Time']
        
        available_columns = [col for col in numeric_columns if col in df_viz.columns]
        
        if len(available_columns) < 3:
            return
        
        # Группируем по моделям и нормализуем данные
        model_stats = df_viz.groupby('Model')[available_columns].mean()
        
        # Нормализация для лучшей визуализации
        normalized_stats = model_stats.copy()
        for col in available_columns:
            if col not in ['Bias_Shift', 'Concept_Drift']:  # Для этих метрик меньше = лучше
                normalized_stats[col] = (model_stats[col] - model_stats[col].min()) / (model_stats[col].max() - model_stats[col].min())
            else:
                normalized_stats[col] = 1 - (model_stats[col] - model_stats[col].min()) / (model_stats[col].max() - model_stats[col].min())
        
        plt.figure(figsize=(14, 10))
        
        # Создаем тепловую карту
        im = plt.imshow(normalized_stats.values, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        
        # Настройки осей
        plt.xticks(range(len(available_columns)), available_columns, rotation=45, ha='right')
        plt.yticks(range(len(model_stats.index)), model_stats.index)
        
        # Добавляем оригинальные значения в ячейки
        for i in range(len(model_stats.index)):
            for j in range(len(available_columns)):
                original_value = model_stats.iloc[i, j]
                color = 'white' if normalized_stats.iloc[i, j] < 0.5 else 'black'
                
                if available_columns[j] == 'Training_Time':
                    text = f'{original_value:.2f}s'
                else:
                    text = f'{original_value:.3f}'
                    
                plt.text(j, i, text, ha='center', va='center',
                        color=color, fontweight='bold', fontsize=9)
        
        plt.title('🌡️ Detailed Heatmap of Trust-ADE Metrics\n(green = better, red = worse)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Metrics', fontweight='bold')
        plt.ylabel('Models', fontweight='bold')
        
        # Цветовая шкала
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Normalized Value (0-1)')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/detailed_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"    ❌ Ошибка в create_detailed_heatmap: {e}")
        plt.close()


# Основная функция для создания всех графиков
def create_fixed_visualizations(df_viz, results_dir, timestamp):
    """Обертка для создания расширенной визуализации (совместимость с существующим кодом)"""
    create_extended_visualizations(df_viz, results_dir, timestamp)

