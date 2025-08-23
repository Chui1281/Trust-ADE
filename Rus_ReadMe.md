# 🚀 Trust-ADE: Протокол динамической оценки доверия к системам ИИ

**Trust-ADE (Trust Assessment through Dynamic Explainability)** — это комплексный протокол для количественной оценки доверия к системам искусственного интеллекта, основанный на научном исследовании "От корреляций к каузальности: подход XAI 2.0 и протокол Trust-ADE для доверенного ИИ".

## 📊 Протокол Trust-ADE

### 🎯 Основная формула

Протокол агрегирует три ключевых измерения в единую метрику доверия:

```
Trust_ADE = w_E × ES + w_R × (RI × e^(-γ × CDR)) + w_F × (1 - BSI)
```

**Где:**

- **ES** - Explainability Score (оценка объяснимости)
- **RI** - Robustness Index (индекс устойчивости)
- **CDR** - Concept-Drift Rate (скорость дрейфа концептов)
- **BSI** - Bias Shift Index (индекс смещения предвзятости)
- **w_E, w_R, w_F** - доменно-специфичные веса
- **γ** - параметр чувствительности к дрейфу концептов


### 🔍 Компоненты протокола

#### 1. Explainability Score (ES)

Оценивает качество объяснений через четыре измерения:

```
ES = w_c × F_c + w_s × C_s + w_i × S_i + w_h × U_h
```

- **F_c** - Каузальная фиделити:

```
F_c = |E_sys ∩ E_exp|/|E_exp| × α + |E_sys ∩ E_exp|/|E_sys| × (1-α)
```

- **C_s** - Семантическая когерентность:

```
C_s = 1 - H(E)/H_max
```

- **S_i** - Стабильность интерпретаций:

```
S_i = 1 - (1/N) × Σ d(E_i, E_i^ε)
```

- **U_h** - Человеческая понятность (экспертная оценка)


#### 2. Robustness Index (RI)

Интегрирует устойчивость к различным типам возмущений:

```
RI = w_a × R_a + w_n × R_n + w_e × R_e
```

- **R_a** - Adversarial устойчивость
- **R_n** - Шумовая устойчивость
- **R_e** - Устойчивость объяснений


#### 3. Concept-Drift Rate (CDR)

Измеряет скорость изменения концептуальных зависимостей:

```
CDR = λ × KS(P_t, P_t-Δt) + (1-λ) × JS(P_t, P_t-Δt)
```

- **KS** - статистика Колмогорова-Смирнова
- **JS** - дивергенция Йенсена-Шеннона


#### 4. Bias Shift Index (BSI)

Отслеживает динамику предвзятостей:

```
BSI = √(w_dp × DP_Δ² + w_eo × EO_Δ² + w_cf × CF_Δ²)
```

- **DP_Δ** - изменение демографического паритета
- **EO_Δ** - изменение равенства шансов
- **CF_Δ** - изменение калиброванной справедливости


## 🏗️ Архитектура проекта

```
trust_ade/
├── 📁 trust_ade/              # Основные модули протокола
│   ├── trust_ade.py           # Главный класс TrustADE  
│   ├── trust_calculator.py    # Вычисление итоговой метрики
│   ├── explainability_score.py # Модуль оценки объяснимости
│   ├── robustness_index.py    # Анализ устойчивости
│   ├── bias_shift_index.py    # Детекция предвзятостей
│   ├── concept_drift.py       # Мониторинг дрейфа концептов
│   ├── base_model.py          # Базовый интерфейс моделей
│   └── utils.py              # Вспомогательные утилиты
|
├── 📁 config/                    # Конфигурация и настройки
│   └── settings.py               # Глобальные настройки, CUDA конфиг
│
├── 📁 models/                 # Интеграции ML моделей
│   ├── sklearn_wrapper.py     # Обертка для Scikit-learn
|   ├── wrappers.py              # Базовые обертки
│   |── cuda_models.py           # CUDA-оптимизированные модели
│   └── __init__.py
│
├── 📁 explainers/             # Модули объяснимости
│   ├── shap_explainer.py     # SHAP интеграция
│   └── __init__.py
│
├── 📁 data/                     # Работа с данными
│   └── datasets.py              # Загрузка и подготовка датасетов
│
├── 📁 training/                 # Обучение моделей
│   └── trainers.py              # Тренировка всех типов моделей
│
├── 📁 evaluation/               # Оценка и Trust-ADE
│   └── trust_evaluator.py      # Trust-ADE протокол оценки
│
├── 📁 visualization/            # Визуализация результатов
│   └── charts.py                # Создание графиков и отчетов
│
├── 📁 utils/                    # Утилиты
│   └── io_utils.py              # Сохранение/загрузка результатов
│
├── 📁 analysis/                 # Анализ результатов
│   └── results.py               # Финальный анализ и сравнение
│
├── 📁 cli/                      # Командная строка
│   └── dataset_selector.py     # CLI для выбора датасетов
│
├── 📄 main.py                   # Главный скрипт запуска
│
├── 📁 tests/                  # Тесты
│   ├── test_basic.py         # Базовые тесты
|   ├── demo_trust_ade.py      # Базовая демонстрация
│   └── test_installation.py  # Проверка установки
│
└── 📁 results/                # Результаты анализа
```


## 📦 Установка

### Системные требования

```bash
Python >= 3.8
NumPy >= 1.21.0
Pandas >= 1.3.0
Scikit-learn >= 1.0.0
```


### Быстрая установка

```bash
git clone https://github.com/your-org/trust-ade.git
cd trust-ade
pip install -r requirements.txt
python setup.py install
```


### Зависимости

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
shap>=0.41.0
scipy>=1.7.0
tqdm>=4.62.0
```

## 🚀 Быстрый старт

### 1. Запуск на всех датасетах

```bash
python main.py
```


### 2. Выборочный запуск

```bash
# Только конкретные датасеты
python main.py --datasets iris breast_cancer

# Исключить большие датасеты
python main.py --exclude digits_binary

# Быстрое тестирование
python main.py --datasets iris wine --quick

# Только CUDA модели
python main.py --cuda-only

# Подробный вывод
python main.py --datasets breast_cancer --verbose
```


### 3. Помощь по командам

```bash
python main.py --help
```


## 🎯 Основные возможности

### ✅ Поддерживаемые модели

- **Random Forest** - Ансамбль решающих деревьев
- **MLP Neural Network (CPU)** - Многослойный персептрон (sklearn)
- **MLP Neural Network (CUDA)** - Оптимизированная PyTorch модель с GPU
- **Support Vector Machine** - Метод опорных векторов
- **Gradient Boosting** - Градиентный бустинг
- **XANFIS** - Adaptive Neuro-Fuzzy система (опционально)


### 📊 Поддерживаемые датасеты

- **Iris** - Классификация ирисов (3 класса, 4 признака)
- **Breast Cancer** - Диагностика рака молочной железы (2 класса, 30 признаков)
- **Wine** - Классификация вин (3 класса, 13 признаков)
- **Digits Binary** - Распознавание цифры 0 (2 класса, 64 пикселя)


### 🔬 Trust-ADE метрики

- **Trust Score** - Итоговая оценка доверия (0-1)
- **Explainability Score** - Качество объяснений
- **Robustness Index** - Устойчивость к возмущениям
- **Bias Shift Index** - Индекс предвзятости
- **Concept Drift Rate** - Скорость дрейфа концептов


## 📈 Результаты работы

### Генерируемые файлы

```
results/
├── detailed_comparison_cuda_20250822_143052.csv    # Подробные результаты
├── summary_comparison_cuda_20250822_143052.csv     # Краткая сводка
├── full_results_cuda_20250822_143052.json         # Полные данные
├── fixed_main_comparison_20250822_143052.png       # Основной график
├── trust_metrics_analysis_fixed_20250822_143052.png # Анализ метрик
├── cuda_performance_detailed_20250822_143052.png   # CUDA vs CPU
└── correlation_analysis_fixed_20250822_143052.png  # Корреляции
```


### Пример вывода

```bash
🎯 ОБЩИЙ РЕЙТИНГ МОДЕЛЕЙ (средний Trust Score):
  🥇 MLP Neural Network (CUDA): 0.847 ± 0.023 (на 4 датасетах) 🚀
  🥈 Random Forest: 0.832 ± 0.019 (на 4 датасетах) 💻
  🥉 Gradient Boosting: 0.798 ± 0.031 (на 4 датасетах) 💻

🚀 ПРОИЗВОДИТЕЛЬНОСТЬ CUDA vs CPU:
  🚀 CUDA модели: Trust Score = 0.847, Время = 2.34s
  💻 CPU модели: Trust Score = 0.815, Время = 8.91s
```

## 📊 Шкала зрелости объяснимости L0-L6

| Уровень | Описание | Trust-ADE Support |
| :-- | :-- | :-- |
| **L0** | Полная непрозрачность | ❌ |
| **L1** | Базовые post-hoc объяснения | ✅ LIME, SHAP |
| **L2** | Улучшенные post-hoc + валидация | ✅ |
| **L3** | Частичная архитектурная прозрачность | ✅ |
| **L4** | Глобальная интерпретируемость | ✅ **Trust-ADE L4** |
| **L5** | Контекстно-адаптивные объяснения | ✅ **Trust-ADE L5** |
| **L6** | Автономные самопоясняющиеся системы | 🚧 В разработке |


## 🛠️ Программное API

### Основное использование

```python
from data.datasets import prepare_datasets, create_models_config
from training.trainers import train_models
from evaluation.trust_evaluator import enhanced_trust_ade_evaluation
from sklearn.model_selection import train_test_split

# Подготовка данных
datasets = prepare_datasets()
models_config = create_models_config()

# Выбор датасета
dataset_name = 'breast_cancer'
dataset_info = datasets[dataset_name]
X, y = dataset_info['X'], dataset_info['y']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Обучение моделей
trained_models = train_models(
    X_train, X_test, y_train, y_test,
    dataset_info['feature_names'], models_config,
    dataset_info['type'], dataset_name
)

# Trust-ADE оценка
enhanced_trust_ade_evaluation(
    trained_models, X_test, y_test, 
    dataset_info['domain'], X_train
)

# Результаты доступны в trained_models[model_name]['trust_results']
```


### Кастомная модель

```python
from models.wrappers import SklearnWrapper
from sklearn.ensemble import ExtraTreesClassifier

# Создание кастомной модели
custom_model = ExtraTreesClassifier(n_estimators=150, random_state=42)
custom_model.fit(X_train, y_train)

# Обертка для Trust-ADE
wrapped_model = SklearnWrapper(
    model=custom_model,
    feature_names=[f"feature_{i}" for i in range(X_train.shape[^1])]
)

# Добавление в trained_models для оценки
trained_models['Custom Extra Trees'] = {
    'wrapped_model': wrapped_model,
    'scaler': None,
    'training_time': 1.23,
    'accuracy': 0.95,
    'needs_scaling': False,
    'description': 'Extra Trees Classifier',
    'color': '#FF6B6B',
    'use_cuda': False
}
```


### Работа с CUDA моделями

```python
from models.cuda_models import OptimizedCUDAMLPClassifier
from models.wrappers import CUDAMLPWrapper

# Создание CUDA модели
cuda_model = OptimizedCUDAMLPClassifier(
    hidden_layers=(128, 64, 32),
    n_classes=len(np.unique(y_train)),
    learning_rate=0.001,
    epochs=200,
    dataset_size=len(X_train)
)

# Обучение
cuda_model.fit(X_train, y_train)

# Обертка
wrapped_cuda = CUDAMLPWrapper(cuda_model, feature_names)
```


## ⚙️ Конфигурация

### Настройка CUDA

```python
# config/settings.py
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
CUDA_EFFICIENT_THRESHOLD = 500  # Минимальный размер для CUDA
```


### Доменные конфигурации

```python
# Веса для различных доменов
DOMAIN_CONFIGS = {
    'medical': {'w_E': 0.5, 'w_R': 0.3, 'w_F': 0.2},
    'financial': {'w_E': 0.3, 'w_R': 0.4, 'w_F': 0.3},
    'general': {'w_E': 0.4, 'w_R': 0.3, 'w_F': 0.3}
}
```


## 📊 Визуализация

Система автоматически генерирует:

1. **Основное сравнение** - Trust Score vs Accuracy
2. **Детальный анализ метрик** - Все Trust-ADE компоненты
3. **CUDA vs CPU сравнение** - Производительность и качество
4. **Корреляционный анализ** - Взаимосвязи между метриками

### Кастомная визуализация

```python
from visualization.charts import create_fixed_visualizations
import pandas as pd

# Подготовка данных для визуализации
viz_data = []
for dataset_name, results in all_results.items():
    for model_name, model_info in results['models'].items():
        trust_results = model_info.get('trust_results', {})
        viz_data.append({
            'Dataset': dataset_name,
            'Model': model_name,
            'Trust_Score': trust_results.get('trust_score', 0),
            'Accuracy': model_info.get('accuracy', 0),
            'CUDA': model_info.get('use_cuda', False)
        })

df_viz = pd.DataFrame(viz_data)
create_fixed_visualizations(df_viz, 'results', '20250822_custom')
```


## 🧪 Тестирование

```bash
# Проверка установки
python tests/test_installation.py

# Базовые тесты
python tests/test_basic.py

# Тест конкретного модуля
python -c "from training.trainers import train_models; print('✅ Trainers OK')"
```


## 🔧 Расширение системы

### Добавление нового датасета

```python
# data/datasets.py
def prepare_datasets():
    datasets = {}
    
    # Ваш кастомный датасет
    datasets['custom_dataset'] = {
        'X': your_X_data,
        'y': your_y_data,
        'feature_names': your_feature_names,
        'target_names': your_target_names,
        'description': 'Описание датасета',
        'domain': 'your_domain',
        'type': 'binary'  # или 'multiclass'
    }
    
    return datasets
```


### Новый тип модели

```python
# models/your_model.py
from models.wrappers import SklearnWrapper

class YourModelWrapper(SklearnWrapper):
    def __init__(self, model, feature_names=None):
        super().__init__(model, feature_names)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
```


## 🚀 Примеры использования

### Медицинская диагностика

```bash
python main.py --datasets breast_cancer --verbose
```


### Финансовый анализ

```bash
python main.py --datasets wine --cuda-only
```


### Быстрое сравнение

```bash
python main.py --datasets iris --quick
```


## 📄 Логи выполнения

Пример подробного лога:

```
🔬 ПРОДВИНУТОЕ СРАВНЕНИЕ ML МОДЕЛЕЙ С TRUST-ADE PROTOCOL + CUDA
================================================================================
📊 ТЕСТИРОВАНИЕ НА ДАТАСЕТЕ: BREAST_CANCER
📋 Описание: Диагностика рака молочной железы (2 класса, 30 признаков)
🏷️ Домен: medical
🎯 Тип задачи: binary
================================================================================

  📈 Обучение Random Forest...
    ✅ Random Forest обучен за 0.12 сек, точность: 0.965

  📈 Обучение MLP Neural Network (CUDA)...
      🚀 Используем CUDA (датасет большой: 398)
      Epoch 0/200, Loss: 0.6891
      Epoch 25/200, Loss: 0.1234
    ✅ MLP Neural Network (CUDA) обучен за 2.34 сек, точность: 0.971
    🚀 Использовалось CUDA ускорение

🔍 Enhanced Trust-ADE оценка моделей...
  📊 Оценка Random Forest...
    🎯 Trust Score: 0.832
    📊 Уровень доверия: Высокий
    📈 Метрики: Bias=0.023, Drift=0.045

📊 РЕЗУЛЬТАТЫ ДЛЯ BREAST_CANCER:
Модель                              Точность   Trust Score  Уровень доверия      CUDA
------------------------------------------------------------------------------------------
MLP Neural Network (CUDA)          0.971      0.847        Высокий              🚀
Random Forest                       0.965      0.832        Высокий              💻
```

## 🔬 Научные основы

Протокол Trust-ADE основан на исследовании:

- **Каузальная интерпретируемость** вместо корреляций
- **Динамический мониторинг** качества объяснений
- **Интегральная оценка** объяснимости, устойчивости и справедливости
- **Соответствие стандартам** ISO/IEC 24029 и EU AI Act


### Формулы компонентов

**Explainability Score:**

```
ES = w_c × F_c + w_s × C_s + w_i × S_i + w_h × U_h

где:
F_c = |E_sys ∩ E_exp|/|E_exp| × α + |E_sys ∩ E_exp|/|E_sys| × (1-α)
C_s = 1 - H(E)/H_max
S_i = 1 - (1/N) × Σ d(E_i, E_i^ε)
```

**Robustness Index:**

```
RI = w_a × R_a + w_n × R_n + w_e × R_e

где:
R_a = 1 - (1/|A|) × Σ I[f(x + a) ≠ f(x)]
R_n = E[similarity(f(x), f(x + ε))]
R_e = E[similarity(E(x), E(x + ε))]
```

**Concept-Drift Rate:**

```
CDR = λ × KS(P_t, P_t-Δt) + (1-λ) × JS(P_t, P_t-Δt)
```

**Bias Shift Index:**

```
BSI = √(w_dp × DP_Δ² + w_eo × EO_Δ² + w_cf × CF_Δ²)
```


## 📄 Цитирование

Если вы используете Trust-ADE в своих исследованиях, пожалуйста, цитируйте:

```bibtex
@article{trofimov2025trust_ade,
  title={От корреляций к каузальности: подход XAI 2.0 и протокол Trust-ADE для доверенного ИИ},
  author={Трофимов, Ю.В. and Аверкин, А.Н. and Ильин, А.С. and Лебедев, А.Д.},
  journal={Научный журнал по объяснимому ИИ},
  year={2025}
}
```

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Создайте Pull Request

## 📄 Лицензия

MIT License - свободное использование для исследований и коммерческих проектов.

***

**Trust-ADE** — Ваш надежный инструмент для создания доверенных систем искусственного интеллекта на основе научно обоснованного протокола динамической оценки объяснимости, устойчивости и справедливости! 🚀🔬

