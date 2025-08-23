<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# 🚀 Trust-ADE: Dynamic Trust Assessment Protocol for AI Systems

**Trust-ADE (Trust Assessment through Dynamic Explainability)** is a comprehensive protocol for quantitative trust assessment of artificial intelligence systems, based on the scientific study "From Correlations to Causality: The XAI 2.0 Approach and the Trust-ADE Protocol for Trustworthy AI".

## 📊 Trust-ADE Protocol

### 🎯 Main Formula

The protocol aggregates three key dimensions into a single trust metric:

```
Trust_ADE = w_E × ES + w_R × (RI × e^(-γ × CDR)) + w_F × (1 - BSI)
```

**Where:**

- **ES** - Explainability Score
- **RI** - Robustness Index
- **CDR** - Concept-Drift Rate
- **BSI** - Bias Shift Index
- **w_E, w_R, w_F** - domain-specific weights
- **γ** - drift sensitivity parameter

***

### 🔍 Protocol Components

#### 1. Explainability Score (ES)

Assesses explanation quality via four dimensions:

```
ES = w_c × F_c + w_s × C_s + w_i × S_i + w_h × U_h
```

- **F_c** - Causal fidelity:

```
F_c = |E_sys ∩ E_exp|/|E_exp| × α + |E_sys ∩ E_exp|/|E_sys| × (1-α)
```

- **C_s** - Semantic coherence:

```
C_s = 1 - H(E)/H_max
```

- **S_i** - Interpretation stability:

```
S_i = 1 - (1/N) × Σ d(E_i, E_i^ε)
```

- **U_h** - Human comprehension (expert assessment)

***

#### 2. Robustness Index (RI)

Aggregates robustness to various perturbations:

```
RI = w_a × R_a + w_n × R_n + w_e × R_e
```

- **R_a** - Adversarial robustness
- **R_n** - Noise robustness
- **R_e** - Explanation robustness

***

#### 3. Concept-Drift Rate (CDR)

Measures the rate of conceptual changes:

```
CDR = λ × KS(P_t, P_t-Δt) + (1-λ) × JS(P_t, P_t-Δt)
```

- **KS** - Kolmogorov-Smirnov statistic
- **JS** - Jensen-Shannon divergence

***

#### 4. Bias Shift Index (BSI)

Tracks dynamics of biases:

```
BSI = √(w_dp × DP_Δ² + w_eo × EO_Δ² + w_cf × CF_Δ²)
```

- **DP_Δ** - Demographic parity change
- **EO_Δ** - Equalized odds change
- **CF_Δ** - Calibrated fairness change

***

## 🏗️ Project Architecture

```
trust_ade/
├── 📁 trust_ade/             # Core protocol modules
│   ├── trust_ade.py          # Main TrustADE class
│   ├── trust_calculator.py   # Metric computation
│   ├── explainability_score.py # Explainability evaluation module
│   ├── robustness_index.py   # Robustness analysis
│   ├── bias_shift_index.py   # Bias detection
│   ├── concept_drift.py      # Concept drift monitoring
│   ├── base_model.py         # Base model interface
│   └── utils.py              # Utilities
|
├── 📁 config/                # Config and settings
│   └── settings.py           # Global settings, CUDA config
│
├── 📁 models/                # ML model integrations
│   ├── sklearn_wrapper.py    # Scikit-learn wrapper
|   ├── wrappers.py           # Basic wrappers
│   ├── cuda_models.py        # CUDA-optimized models
│   └── __init__.py
│
├── 📁 explainers/            # Explainability modules
│   ├── shap_explainer.py     # SHAP integration
│   └── __init__.py
│
├── 📁 data/                  # Data utilities
│   └── datasets.py           # Dataset loading and prepping
│
├── 📁 training/              # Model training
│   └── trainers.py           # Trainers for all model types
│
├── 📁 evaluation/            # Evaluation and Trust-ADE
│   └── trust_evaluator.py    # Trust-ADE assessment protocol
│
├── 📁 visualization/         # Visualization
│   └── charts.py             # Plots and report generation
│
├── 📁 utils/                 # Utilities
│   └── io_utils.py           # Save/load results
│
├── 📁 analysis/              # Result analysis
│   └── results.py            # Final analysis and comparison
│
├── 📁 cli/                   # Command-line interface
│   └── dataset_selector.py   # CLI for dataset selection
│
├── 📄 main.py                # Main run script
│
├── 📁 tests/                 # Tests
│   ├── test_basic.py         # Basic tests
|   ├── demo_trust_ade.py     # Demo
│   └── test_installation.py  # Installation test
│
└── 📁 results/               # Analysis results
```


***

## 📦 Installation

### System requirements

```bash
Python >= 3.8
NumPy >= 1.21.0
Pandas >= 1.3.0
Scikit-learn >= 1.0.0
```


### Quick install

```bash
git clone https://github.com/your-org/trust-ade.git
cd trust-ade
pip install -r requirements.txt
python setup.py install
```


### Dependencies

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


***

## 🚀 Getting Started

### 1. Run on all datasets

```bash
python main.py
```


### 2. Selective run

```bash
# Only specific datasets
python main.py --datasets iris breast_cancer

# Exclude large datasets
python main.py --exclude digits_binary

# Quick test
python main.py --datasets iris wine --quick

# CUDA models only
python main.py --cuda-only

# Verbose output
python main.py --datasets breast_cancer --verbose
```


### 3. Help on commands

```bash
python main.py --help
```


***

## 🎯 Key features

### ✅ Supported models

- **Random Forest** - Ensemble of decision trees
- **MLP Neural Network (CPU)** - Multilayer perceptron (sklearn)
- **MLP Neural Network (CUDA)** - Optimized PyTorch model with GPU
- **Support Vector Machine**
- **Gradient Boosting**
- **XANFIS** - Adaptive Neuro-Fuzzy System (optional)


### 📊 Supported datasets

- **Iris** - Iris classification (3 classes, 4 features)
- **Breast Cancer** - Breast cancer diagnosis (2 classes, 30 features)
- **Wine** - Wine classification (3 classes, 13 features)
- **Digits Binary** - Zero vs other digit recognition (2 classes, 64 pixels)


### 🔬 Trust-ADE Metrics

- **Trust Score** - Overall confidence (0-1)
- **Explainability Score** - Explanation quality
- **Robustness Index** - Robustness to perturbations
- **Bias Shift Index** - Bias index
- **Concept Drift Rate** - Rate of concept drift

***

## 📈 Output

### Generated files

```
results/
├── detailed_comparison_cuda_20250822_143052.csv    # Detailed results
├── summary_comparison_cuda_20250822_143052.csv     # Summary
├── full_results_cuda_20250822_143052.json          # Full data
├── fixed_main_comparison_20250822_143052.png       # Main plot
├── trust_metrics_analysis_fixed_20250822_143052.png # Metric analysis
├── cuda_performance_detailed_20250822_143052.png   # CUDA vs CPU
└── correlation_analysis_fixed_20250822_143052.png  # Correlations
```


### Example output

```bash
🎯 OVERALL MODEL RANKING (average Trust Score):
  🥇 MLP Neural Network (CUDA): 0.847 ± 0.023 (over 4 datasets) 🚀
  🥈 Random Forest: 0.832 ± 0.019 (over 4 datasets) 💻
  🥉 Gradient Boosting: 0.798 ± 0.031 (over 4 datasets) 💻

🚀 CUDA VS CPU PERFORMANCE:
  🚀 CUDA models: Trust Score = 0.847, Time = 2.34s
  💻 CPU models: Trust Score = 0.815, Time = 8.91s
```


***

## 📊 Explainability Maturity Scale L0-L6

| Level | Description | Trust-ADE Support |
| :-- | :-- | :-- |
| **L0** | Full opacity | ❌ |
| **L1** | Basic post-hoc explanations | ✅ LIME, SHAP |
| **L2** | Improved post-hoc + validation | ✅ |
| **L3** | Partial architectural transparency | ✅ |
| **L4** | Global interpretability | ✅ **Trust-ADE L4** |
| **L5** | Context-adaptive explanations | ✅ **Trust-ADE L5** |
| **L6** | Autonomous self-explaining systems | 🚧 In development |


***

## 🛠️ Programmatic API

### Basic usage

```python
from data.datasets import prepare_datasets, create_models_config
from training.trainers import train_models
from evaluation.trust_evaluator import enhanced_trust_ade_evaluation
from sklearn.model_selection import train_test_split

# Prepare data
datasets = prepare_datasets()
models_config = create_models_config()

# Select dataset
dataset_name = 'breast_cancer'
dataset_info = datasets[dataset_name]
X, y = dataset_info['X'], dataset_info['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train models
trained_models = train_models(
    X_train, X_test, y_train, y_test,
    dataset_info['feature_names'], models_config,
    dataset_info['type'], dataset_name
)

# Trust-ADE evaluation
enhanced_trust_ade_evaluation(
    trained_models, X_test, y_test,
    dataset_info['domain'], X_train
)

# Results available at trained_models[model_name]['trust_results']
```


### Custom model

```python
from models.wrappers import SklearnWrapper
from sklearn.ensemble import ExtraTreesClassifier

# Create custom model
custom_model = ExtraTreesClassifier(n_estimators=150, random_state=42)
custom_model.fit(X_train, y_train)

# Wrap for Trust-ADE
wrapped_model = SklearnWrapper(
    model=custom_model,
    feature_names=[f"feature_{i}" for i in range(X_train.shape[^1])]
)

# Add to trained_models for evaluation
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


### Working with CUDA models

```python
from models.cuda_models import OptimizedCUDAMLPClassifier
from models.wrappers import CUDAMLPWrapper

# Create CUDA model
cuda_model = OptimizedCUDAMLPClassifier(
    hidden_layers=(128, 64, 32),
    n_classes=len(np.unique(y_train)),
    learning_rate=0.001,
    epochs=200,
    dataset_size=len(X_train)
)

# Train
cuda_model.fit(X_train, y_train)

# Wrap
wrapped_cuda = CUDAMLPWrapper(cuda_model, feature_names)
```


***

## ⚙️ Configuration

### CUDA setup

```python
# config/settings.py
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
CUDA_EFFICIENT_THRESHOLD = 500  # Min size for CUDA
```


### Domain configurations

```python
# Weights for different domains
DOMAIN_CONFIGS = {
    'medical': {'w_E': 0.5, 'w_R': 0.3, 'w_F': 0.2},
    'financial': {'w_E': 0.3, 'w_R': 0.4, 'w_F': 0.3},
    'general': {'w_E': 0.4, 'w_R': 0.3, 'w_F': 0.3}
}
```


***

## 📊 Visualization

The system auto-generates:

1. **Main comparison** - Trust Score vs Accuracy
2. **Detailed metrics analysis** - All Trust-ADE components
3. **CUDA vs CPU comparison** - Performance and quality
4. **Correlation analysis** - Metric relationships

### Custom visualization

```python
from visualization.charts import create_fixed_visualizations
import pandas as pd

# Prepare data for visualization
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


***

## 🧪 Testing

```bash
# Installation test
python tests/test_installation.py

# Basic tests
python tests/test_basic.py

# Test a module
python -c "from training.trainers import train_models; print('✅ Trainers OK')"
```


***

## 🔧 System Extension

### Add new dataset

```python
# data/datasets.py
def prepare_datasets():
    datasets = {}
    
    # Your custom dataset
    datasets['custom_dataset'] = {
        'X': your_X_data,
        'y': your_y_data,
        'feature_names': your_feature_names,
        'target_names': your_target_names,
        'description': 'Dataset description',
        'domain': 'your_domain',
        'type': 'binary'  # or 'multiclass'
    }
    
    return datasets
```


### New model type

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


***

## 🚀 Usage Examples

### Medical diagnosis

```bash
python main.py --datasets breast_cancer --verbose
```


### Financial analysis

```bash
python main.py --datasets wine --cuda-only
```


### Quick comparison

```bash
python main.py --datasets iris --quick
```


***

## 📄 Logs

Sample detailed log:

```
🔬 ADVANCED ML MODEL COMPARISON WITH TRUST-ADE PROTOCOL + CUDA
================================================================================
📊 TESTING ON DATASET: BREAST_CANCER
📋 Description: Breast cancer diagnosis (2 classes, 30 features)
🏷️ Domain: medical
🎯 Task type: binary
================================================================================

  📈 Training Random Forest...
    ✅ Random Forest trained in 0.12 sec, accuracy: 0.965

  📈 Training MLP Neural Network (CUDA)...
      🚀 Using CUDA (large dataset: 398)
      Epoch 0/200, Loss: 0.6891
      Epoch 25/200, Loss: 0.1234
    ✅ MLP Neural Network (CUDA) trained in 2.34 sec, accuracy: 0.971
    🚀 CUDA acceleration used

🔍 Enhanced Trust-ADE model assessment...
  📊 Assessing Random Forest...
    🎯 Trust Score: 0.832
    📊 Trust level: High
    📈 Metrics: Bias=0.023, Drift=0.045

📊 BREAST_CANCER RESULTS:
Model                              Accuracy   Trust Score  Trust Level         CUDA
------------------------------------------------------------------------------------------
MLP Neural Network (CUDA)          0.971      0.847        High               🚀
Random Forest                      0.965      0.832        High               💻
```


***

## 🔬 Scientific Foundations

Trust-ADE is based on the research:

- **Causal interpretability** instead of mere correlations
- **Dynamic monitoring** of explanation quality
- **Integral assessment** of explainability, robustness, and fairness
- **Compliance** with ISO/IEC 24029 and the EU AI Act


### Component formulas

**Explainability Score:**

```
ES = w_c × F_c + w_s × C_s + w_i × S_i + w_h × U_h

where:
F_c = |E_sys ∩ E_exp|/|E_exp| × α + |E_sys ∩ E_exp|/|E_sys| × (1-α)
C_s = 1 - H(E)/H_max
S_i = 1 - (1/N) × Σ d(E_i, E_i^ε)
```

**Robustness Index:**

```
RI = w_a × R_a + w_n × R_n + w_e × R_e

where:
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


***

## 📄 Citation

If you use Trust-ADE in your research, please cite:

```bibtex
@article{trofimov2025trust_ade,
  title={From Correlations to Causality: The XAI 2.0 Approach and the Trust-ADE Protocol for Trustworthy AI},
  author={Trofimov, Y.V. and Averkin, A.N. and Ilyin, A.S. and Lebedev, A.D.},
  journal={Journal of Explainable AI},
  year={2025}
}
```


***

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to your branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

***

## 📄 License

MIT License - free to use for research and commercial projects.

***

**Trust-ADE** — Your reliable tool for building trustworthy artificial intelligence systems based on a scientifically grounded protocol for dynamic explainability, robustness, and fairness assessment! 🚀🔬

<div style="text-align: center">⁂</div>

[^1]: paste.txt

