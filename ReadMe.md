# 📊 Trust-ADE: Advanced Model Evaluation with CUDA Support

**Trust-ADE (Trustworthy AI Decision Engine)** - это передовая система для комплексной оценки доверия к моделям машинного обучения с поддержкой GPU ускорения и интеграцией современных ML фреймворков.

## 🚀 Основные возможности

### ✨ **Trust-ADE Protocol**

- 📈 **Многомерная оценка доверия**: Trust Score, объяснимость, устойчивость
- 🔍 **Детекция смещений**: Bias shift detection и concept drift analysis
- 🎯 **Адаптивные метрики**: Автоматическая настройка под тип задачи
- 📊 **Профессиональная визуализация**: Интерактивные графики и отчеты


### 🚀 **CUDA Оптимизация**

- ⚡ **Адаптивное GPU ускорение**: Автоматический выбор CPU/CUDA
- 📱 **Интеллектуальное переключение**: CPU для малых данных, GPU для больших
- 🔧 **Оптимизированные архитектуры**: Динамическое масштабирование нейросетей
- 💾 **Эффективное использование памяти**: Управление GPU ресурсами


### 🤖 **Поддержка моделей**

- 🌲 **Scikit-learn**: RF, SVM, MLP, Gradient Boosting и др.
- 🔥 **PyTorch**: Кастомные нейросети с CUDA поддержкой
- 🧠 **XANFIS**: Adaptive Neuro-Fuzzy Inference Systems
- 🔌 **Расширяемость**: Простая интеграция новых моделей


## 📦 Установка

### Быстрая установка

```bash
git clone https://github.com/your-username/Trust-ADE.git
cd Trust-ADE
pip install -r requirements.txt
```


### Требования

```txt
torch>=2.0.0
scikit-learn>=1.3.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.64.0
xanfis>=1.0.0  # опционально
mealpy>=3.0.0  # для XANFIS
permetrics>=1.4.0  # для XANFIS
```


### CUDA поддержка

```bash
# Для NVIDIA GPU с CUDA 11.8+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


## 🎯 Быстрый старт

### Простой пример

```python
from trust_ade.trust_ade import TrustADE
from models.sklearn_wrapper import SklearnWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Загрузка данных
X, y = load_iris(return_X_y=True)

# Создание и обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Обертка модели для Trust-ADE
wrapped_model = SklearnWrapper(model, feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Trust-ADE оценка
trust_evaluator = TrustADE(model=wrapped_model, domain='general')
results = trust_evaluator.evaluate(X, y)

print(f"Trust Score: {results['trust_score']:.3f}")
print(f"Trust Level: {results['trust_level']}")
```


### Полное сравнение моделей с CUDA

```python
# Запуск полного анализа
python Trust-ADE/examples/complete_model_comparison_fixed.py
```


## 📊 Результаты анализа

После запуска полного сравнения вы получите:

### 📁 Файлы результатов

- **`detailed_comparison_cuda_YYYYMMDD_HHMMSS.csv`** - Подробные результаты по всем моделям и датасетам
- **`summary_comparison_cuda_YYYYMMDD_HHMMSS.csv`** - Сводная таблица со средними значениями
- **`full_results_cuda_YYYYMMDD_HHMMSS.json`** - Полные данные в JSON формате


### 🎨 Визуализация

- **`fixed_main_comparison_YYYYMMDD_HHMMSS.png`** - Основное сравнение моделей
- **`trust_metrics_analysis_fixed_YYYYMMDD_HHMMSS.png`** - Детальный анализ метрик Trust-ADE
- **`cuda_performance_detailed_YYYYMMDD_HHMMSS.png`** - CUDA vs CPU сравнение
- **`correlation_analysis_fixed_YYYYMMDD_HHMMSS.png`** - Корреляционный анализ


## 🏆 Примеры результатов

### Лучшие модели по Trust Score:

1. **🥇 Support Vector Machine**: 0.849 ± 0.011
2. **🥈 MLP Neural Network (CPU)**: 0.830 ± 0.031
3. **🥉 MLP Neural Network (CUDA)**: 0.672 ± 0.066

### Тестируемые датасеты:

- 🌸 **Iris Dataset** - Классификация ирисов (3 класса, 4 признака)
- 🏥 **Breast Cancer** - Диагностика рака (2 класса, 30 признаков)
- 🍷 **Wine Dataset** - Классификация вин (3 класса, 13 признаков)
- 🔢 **Digits Dataset** - Распознавание цифр (2 класса, 64 пикселя)


## 🔧 Архитектура проекта

```
Trust-ADE/
├── 📁 trust_ade/           # Основной модуль Trust-ADE
│   ├── trust_ade.py        # Главный класс TrustADE
│   └── __init__.py
├── 📁 models/              # Обертки для моделей
│   ├── sklearn_wrapper.py  # Scikit-learn интеграция
│   └── __init__.py
├── 📁 examples/            # Примеры использования
│   ├── complete_model_comparison_fixed.py  # Полное сравнение
│   └── advanced_model_comparioson.py       # Расширенный анализ
├── 📁 results/             # Результаты анализа
│   ├── *.csv              # CSV отчеты
│   ├── *.json             # JSON данные
│   └── *.png              # Графики
├── 📄 requirements.txt     # Зависимости
└── 📖 README.md           # Документация
```


## ⚙️ Конфигурация

### CUDA настройки

```python
# В скрипте автоматически определяется:
CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_EFFICIENT_THRESHOLD = 500  # Мин. размер для GPU

# Для принудительного использования CPU:
DEVICE = 'cpu'
```


### Trust-ADE параметры

```python
trust_evaluator = TrustADE(
    model=wrapped_model,
    domain='general',  # 'medical', 'general', 'financial'
    training_data=X_train,  # для лучших метрик
    verbose=True  # детальный вывод
)
```


## 🐛 Исправленные проблемы

### ✅ Версия 2.0 исправления:

- **XANFIS интеграция** - Корректные параметры AnfisClassifier
- **Numpy сериализация** - Исправлена JSON совместимость
- **Trust-ADE метрики** - Обработка нулевых значений
- **CUDA оптимизация** - Адаптивный выбор устройства
- **Визуализация** - Исправлены matplotlib colormap ошибки


## 🤝 Расширение системы

### Добавление новой модели

```python
from models.sklearn_wrapper import SklearnWrapper

class CustomModelWrapper(SklearnWrapper):
    def __init__(self, model, feature_names=None):
        super().__init__(model, feature_names)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
```


### Кастомные метрики Trust-ADE

```python
# В trust_ade.py можно добавить новые метрики
def custom_trust_metric(self, y_true, y_pred):
    # Ваша кастомная логика
    return metric_value
```


## 📈 Performance

### Скорость обработки:

- **CPU модели**: ~0.1-2 сек на датасет (150-1500 образцов)
- **CUDA модели**: Автоматическая оптимизация по размеру данных
- **Trust-ADE оценка**: ~1-3 сек на модель
- **Визуализация**: ~2-5 сек на граф


### Память:

- **RAM**: ~1-2 GB для полного анализа
- **VRAM**: ~1-4 GB при использовании CUDA (адаптивно)


## 🎓 Документация и примеры

### Основные классы:

- **`TrustADE`** - Главный класс для оценки доверия
- **`SklearnWrapper`** - Обертка для scikit-learn моделей
- **`OptimizedCUDAMLPClassifier`** - CUDA-оптимизированная нейросеть
- **`FixedXANFISWrapper`** - Исправленная XANFIS интеграция


### Ключевые метрики:

- **Trust Score** - Общая оценка доверия (0-1)
- **Explainability Score** - Объяснимость модели (0-1)
- **Robustness Index** - Устойчивость к возмущениям (0-1)
- **Bias Shift Index** - Детекция смещений (0-1)
- **Concept Drift Rate** - Скорость дрейфа концептов (0-1)


## 📄 Лицензия

MIT License - свободное использование для исследований и коммерческих проектов.

## 🔗 Связь и поддержка

- **Issues**: Создавайте issues для багов и предложений
- **Discussions**: Обсуждение идей и улучшений
- **Wiki**: Подробная документация и туториалы

***

**Trust-ADE** - Ваш надежный инструмент для оценки и сравнения ML моделей! 🚀🔬

