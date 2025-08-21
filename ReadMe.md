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
│
├── 📁 models/                 # Интеграции ML моделей
│   ├── sklearn_wrapper.py     # Обертка для Scikit-learn
│   └── __init__.py
│
├── 📁 explainers/             # Модули объяснимости
│   ├── shap_explainer.py     # SHAP интеграция
│   └── __init__.py
│
├── 📁 examples/               # Примеры использования
│   ├── demo_trust_ade.py      # Базовая демонстрация
│   └── advanced_model_comparison.py  # Сравнение моделей
│
├── 📁 tests/                  # Тесты
│   ├── test_basic.py         # Базовые тесты
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

### Базовое использование

```python
from trust_ade import TrustADE
from models.sklearn_wrapper import SklearnWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Загрузка данных
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Обертка модели для Trust-ADE
wrapped_model = SklearnWrapper(
    model=model,
    feature_names=[f'feature_{i}' for i in range(X.shape[^1])]
)

# Инициализация Trust-ADE
trust_evaluator = TrustADE(
    model=wrapped_model,
    domain='medical',          # медицинский домен
    training_data=X_train,
    verbose=True
)

# Выполнение оценки Trust-ADE
results = trust_evaluator.evaluate(X_test, y_test)

# Результаты
print(f"🎯 Trust Score: {results['trust_score']:.3f}")
print(f"📊 Trust Level: {results['trust_level']}")
print(f"🔍 Explainability Score: {results.get('explainability_score', 'N/A')}")
print(f"🛡️ Robustness Index: {results.get('robustness_index', 'N/A')}")
print(f"📈 Concept Drift Rate: {results.get('concept_drift_rate', 'N/A')}")
print(f"⚖️ Bias Shift Index: {results.get('bias_shift_index', 'N/A')}")
```


### Сравнение моделей

```python
# Запуск полного сравнения
python examples/advanced_model_comparison.py

# Демонстрация базовых функций
python examples/demo_trust_ade.py
```


## 🎯 Доменно-специфичная настройка

### Конфигурации по доменам

```python
DOMAIN_CONFIGS = {
    'medical': {
        'w_E': 0.5,  # Максимальный приоритет объяснимости
        'w_R': 0.3,  # Умеренная важность устойчивости  
        'w_F': 0.2,  # Базовая важность справедливости
        'gamma': 2.0,  # Высокая чувствительность к дрейфу
    },
    'financial': {
        'w_E': 0.3,  # Сбалансированная объяснимость
        'w_R': 0.4,  # Высокий приоритет устойчивости
        'w_F': 0.3,  # Важность справедливости  
        'gamma': 1.5,  # Средняя чувствительность к дрейфу
    },
    'criminal_justice': {
        'w_E': 0.3,  # Умеренная объяснимость
        'w_R': 0.2,  # Низкий приоритет устойчивости
        'w_F': 0.5,  # Максимальная важность справедливости
        'gamma': 2.5,  # Очень высокая чувствительность
    },
    'general': {
        'w_E': 0.4,  # Сбалансированные веса
        'w_R': 0.3,
        'w_F': 0.3,
        'gamma': 1.0,
    }
}
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

## ⚙️ API Документация

### Основные классы

#### TrustADE

```python
class TrustADE:
    def __init__(self, model, domain='general', training_data=None, verbose=False):
        """
        Инициализация системы оценки доверия Trust-ADE
        
        Args:
            model: Обернутая ML модель
            domain: Домен применения ('medical', 'financial', 'criminal_justice', 'general')
            training_data: Тренировочные данные для baseline метрик
            verbose: Детальный вывод процесса оценки
        """
        
    def evaluate(self, X_test, y_test):
        """
        Комплексная оценка доверия по протоколу Trust-ADE
        
        Returns:
            dict: Результаты оценки с детальными метриками
        """
        
    def explain_predictions(self, X, method='shap'):
        """Генерация объяснений для предсказаний"""
        
    def monitor_drift(self, X_new, window_size=100):
        """Мониторинг concept drift в реальном времени"""
```


#### TrustCalculator

```python
class TrustCalculator:
    def calculate_trust_score(self, explainability_score, robustness_index, 
                             bias_shift_index, concept_drift_rate, verbose=False):
        """
        Вычисление итогового Trust-ADE Score
        
        Returns:
            dict: Детальные результаты с компонентами доверия
        """
        
    def adaptive_weight_calibration(self, expert_ratings, computed_scores):
        """Адаптивная калибровка весов по экспертным оценкам"""
        
    def get_recommendations(self, trust_results):
        """Получение рекомендаций по улучшению доверия"""
```


## 🧪 Тестирование

```bash
# Запуск всех тестов
python -m pytest tests/ -v

# Проверка установки
python tests/test_installation.py

# Базовые тесты
python tests/test_basic.py
```


## 📈 Результаты и визуализация

После выполнения оценки система генерирует:

- **CSV отчеты** с детальными метриками
- **JSON файлы** с полными данными
- **Графики** Trust-ADE компонентов
- **Рекомендации** по улучшению доверия

Пример результатов:

```
🎯 Trust Score: 0.847
📊 Trust Level: Высокое доверие
🔍 Explainability Score: 0.782
🛡️ Robustness Index: 0.891  
📈 Concept Drift Rate: 0.124
⚖️ Bias Shift Index: 0.156
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


## 🤝 Расширение системы

### Добавление новой модели

```python
from trust_ade.base_model import BaseModel

class CustomModelWrapper(BaseModel):
    def __init__(self, model, feature_names=None):
        super().__init__(model, feature_names)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
        
    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
```


### Кастомные метрики

```python
class CustomTrustCalculator(TrustCalculator):
    def calculate_custom_explainability(self, model, X, y):
        """Кастомная логика оценки объяснимости"""
        # Ваша реализация
        return explainability_score
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

## 📄 Лицензия

MIT License - свободное использование для исследований и коммерческих проектов.

***

**Trust-ADE** — Ваш надежный инструмент для создания доверенных систем искусственного интеллекта на основе научно обоснованного протокола динамической оценки объяснимости, устойчивости и справедливости! 🚀🔬

