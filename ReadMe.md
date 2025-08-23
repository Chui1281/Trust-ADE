# 🚀 Trust-ADE: Dynamic Trust Assessment Protocol for AI Systems

**Trust-ADE (Trust Assessment through Dynamic Explainability)** is a comprehensive protocol for quantitative trust assessment of artificial intelligence systems, based on the scientific research "From Correlations to Causality: XAI 2.0 Approach and Trust-ADE Protocol for Trustworthy AI".

## 📊 Trust-ADE Protocol

### 🎯 Core Formula

The protocol aggregates three key dimensions into a unified trust metric:

```
Trust_ADE = w_E × ES + w_R × (RI × e^(-γ × CDR)) + w_F × (1 - BSI)
```

**Where:**

- **ES** - Explainability Score
- **RI** - Robustness Index
- **CDR** - Concept-Drift Rate
- **BSI** - Bias Shift Index
- **w_E, w_R, w_F** - domain-specific weights
- **γ** - concept drift sensitivity parameter


### 🔍 Protocol Components

#### 1. Explainability Score (ES)

Evaluates explanation quality through four dimensions:

```
ES = w_c × F_c + w_s × C_s + w_i × S_i + w_h × U_h
```

- **F_c** - Causal Fidelity:

```
F_c = |E_sys ∩ E_exp|/|E_exp| × α + |E_sys ∩ E_exp|/|E_sys| × (1-α)
```

- **C_s** - Semantic Coherence:

```
C_s = 1 - H(E)/H_max
```

- **S_i** - Interpretation Stability:

```
S_i = 1 - (1/N) × Σ d(E_i, E_i^ε)
```

- **U_h** - Human Comprehensibility (expert assessment)


#### 2. Robustness Index (RI)

Integrates robustness to various types of perturbations:

```
RI = w_a × R_a + w_n × R_n + w_e × R_e
```

- **R_a** - Adversarial robustness
- **R_n** - Noise robustness
- **R_e** - Explanation robustness


#### 3. Concept-Drift Rate (CDR)

Measures the rate of conceptual dependency changes:

```
CDR = λ × KS(P_t, P_t-Δt) + (1-λ) × JS(P_t, P_t-Δt)
```

- **KS** - Kolmogorov-Smirnov statistic
- **JS** - Jensen-Shannon divergence


#### 4. Bias Shift Index (BSI)

Tracks bias dynamics:

```
BSI = √(w_dp × DP_Δ² + w_eo × EO_Δ² + w_cf × CF_Δ²)
```

- **DP_Δ** - demographic parity change
- **EO_Δ** - equalized odds change
- **CF_Δ** - calibrated fairness change


## 🏗️ Project Architecture

```
trust_ade/
├── 📁 trust_ade/              # Core protocol modules
│   ├── trust_ade.py           # Main TrustADE class  
│   ├── trust_calculator.py    # Final metric calculation
│   ├── explainability_score.py # Explainability assessment module
│   ├── robustness_index.py    # Robustness analysis
│   ├── bias_shift_index.py    # Bias detection
│   ├── concept_drift.py       # Concept drift monitoring
│   ├── base_model.py          # Base model interface
│   └── utils.py              # Helper utilities
|
├── 📁 config/                    # Configuration and settings
│   └── settings.py               # Global settings, CUDA config
│
├── 📁 models/                 # ML model integrations
│   ├── sklearn_wrapper.py     # Scikit-learn wrapper
|   ├── wrappers.py              # Base wrappers
│   |── cuda_models.py           # CUDA-optimized models
│   └── __init__.py
│
├── 📁 explainers/             # Explainability modules
│   ├── shap_explainer.py     # SHAP integration
│   └── __init__.py
│
├── 📁 data/                     # Data handling
│   └── datasets.py              # Dataset loading and preparation
│
├── 📁 training/                 # Model training
│   └── trainers.py              # Training for all model types
│
├── 📁 evaluation/               # Evaluation and Trust-ADE
│   └── trust_evaluator.py      # Trust-ADE assessment protocol
│
├── 📁 visualization/            # Results visualization
│   └── charts.py                # Chart and report generation
│
├── 📁 utils/                    # Utilities
│   └── io_utils.py              # Save/load results
│
├── 📁 analysis/                 # Results analysis
│   └── results.py               # Final analysis and comparison
│
├── 📁 cli/                      # Command line interface
│   └── dataset_selector.py     # CLI for dataset selection
│
├── 📄 main.py                   # Main execution script
│
├── 📁 tests/                  # Tests
│   ├── test_basic.py         # Basic tests
|   ├── demo_trust_ade.py      # Basic demonstration
│   └── test_installation.py  # Installation verification
│
└── 📁 results/                # Analysis results
```


## 📦 Installation

### System Requirements

```bash
Python >= 3.8
NumPy >= 1.21.0
Pandas >= 1.3.0
Scikit-learn >= 1.0.0
```


### Quick Installation

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


## 🚀 Quick Start

### 1. Run on All Datasets

```bash
python main.py
```


### 2. Selective Execution

```bash
# Specific datasets only
python main.py --datasets iris breast_cancer

# Exclude large datasets
python main.py --exclude digits_binary

# Quick testing
python main.py --datasets iris wine --quick

# CUDA models only
python main.py --cuda-only

# Verbose output
python main.py --datasets breast_cancer --verbose
```


### 3. Command Help

```bash
python main.py --help
```


## 🎯 Key Features

### ✅ Supported Models

- **XANFIS** - Adaptive Neuro-Fuzzy Inference System (🏆 **Best Trust Score**)
- **MLP Neural Network (CPU)** - Multi-layer perceptron (sklearn)
- **Support Vector Machine** - Support vector method
- **Gradient Boosting** - Gradient boosting ensemble
- **Random Forest** - Decision tree ensemble
- **MLP Neural Network (CUDA)** - Optimized PyTorch model with GPU


### 📊 Supported Datasets

- **Iris** - Iris classification (3 classes, 4 features)
- **Breast Cancer** - Breast cancer diagnosis (2 classes, 30 features)
- **Wine** - Wine classification (3 classes, 13 features)
- **Digits Binary** - Digit 0 recognition (2 classes, 64 pixels)


### 🔬 Trust-ADE Metrics

- **Trust Score** - Final trust assessment (0-1)
- **Explainability Score** - Explanation quality
- **Robustness Index** - Perturbation resistance
- **Bias Shift Index** - Bias index
- **Concept Drift Rate** - Concept drift rate


## 📈 Real Performance Results

### Benchmark Results (4 Datasets)

Based on comprehensive testing across iris, breast_cancer, wine, and digits_binary datasets:

```bash
🎯 OVERALL MODEL RANKING (average Trust Score):
  🥇 XANFIS: 0.842 ± 0.078 (on 4 datasets) 💻
  🥈 MLP Neural Network (CPU): 0.584 ± 0.046 (on 4 datasets) 💻
  🥉 Support Vector Machine: 0.562 ± 0.050 (on 4 datasets) 💻
  4️⃣ Gradient Boosting: 0.532 ± 0.099 (on 4 datasets) 💻
  5️⃣ Random Forest: 0.517 ± 0.094 (on 4 datasets) 💻
  6️⃣ MLP Neural Network (CUDA): 0.452 ± 0.099 (on 4 datasets) 💻
```


### Key Findings

- **🏆 XANFIS** achieves the highest Trust Score (0.842) due to superior explainability through rule-based reasoning
- **📊 Explainability vs Accuracy Trade-off**: Higher accuracy doesn't always mean higher trust
- **🔍 CUDA Performance**: Interestingly, CUDA acceleration showed lower trust scores, indicating potential optimization areas
- **📈 Consistency**: XANFIS shows good stability across different domains (±0.078 std deviation)


### Generated Files

```
results/
├── detailed_comparison_cuda_20250823_140323.csv    # Detailed results
├── summary_comparison_cuda_20250823_140323.csv     # Brief summary
├── full_results_cuda_20250823_140323.json         # Complete data
├── fixed_main_comparison_20250823_140323.png       # Main chart
├── trust_metrics_analysis_fixed_20250823_140323.png # Metrics analysis
├── cuda_performance_detailed_20250823_140323.png   # CUDA vs CPU
└── correlation_analysis_fixed_20250823_140323.png  # Correlations
```


## 📊 Explainability Maturity Scale L0-L6

| Level | Description | Trust-ADE Support | XANFIS Achievement |
| :-- | :-- | :-- | :-- |
| **L0** | Complete opacity | ❌ | ❌ |
| **L1** | Basic post-hoc explanations | ✅ LIME, SHAP | ❌ |
| **L2** | Enhanced post-hoc + validation | ✅ | ❌ |
| **L3** | Partial architectural transparency | ✅ | ❌ |
| **L4** | Global interpretability | ✅ **Trust-ADE L4** | ✅ **XANFIS L4-L5** |
| **L5** | Context-adaptive explanations | ✅ **Trust-ADE L5** | ✅ **XANFIS L4-L5** |
| **L6** | Autonomous self-explaining systems | 🚧 In development | 🚧 Future work |

## 🛠️ Programming API

### Basic Usage

```python
from data.datasets import prepare_datasets, create_models_config
from training.trainers import train_models
from evaluation.trust_evaluator import enhanced_trust_ade_evaluation
from sklearn.model_selection import train_test_split

# Data preparation
datasets = prepare_datasets()
models_config = create_models_config()

# Dataset selection
dataset_name = 'breast_cancer'
dataset_info = datasets[dataset_name]
X, y = dataset_info['X'], dataset_info['y']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Model training
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

# Results available in trained_models[model_name]['trust_results']
```


### Custom Model

```python
from models.wrappers import SklearnWrapper
from sklearn.ensemble import ExtraTreesClassifier

# Create custom model
custom_model = ExtraTreesClassifier(n_estimators=150, random_state=42)
custom_model.fit(X_train, y_train)

# Wrapper for Trust-ADE
wrapped_model = SklearnWrapper(
    model=custom_model,
    feature_names=[f"feature_{i}" for i in range(X_train.shape[1])]
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


### Working with CUDA Models

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

# Training
cuda_model.fit(X_train, y_train)

# Wrapper
wrapped_cuda = CUDAMLPWrapper(cuda_model, feature_names)
```


## ⚙️ Configuration

### CUDA Setup

```python
# config/settings.py
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
CUDA_EFFICIENT_THRESHOLD = 500  # Minimum size for CUDA
```


### Domain Configurations

```python
# Weights for different domains
DOMAIN_CONFIGS = {
    'medical': {'w_E': 0.5, 'w_R': 0.3, 'w_F': 0.2},
    'financial': {'w_E': 0.3, 'w_R': 0.4, 'w_F': 0.3},
    'general': {'w_E': 0.4, 'w_R': 0.3, 'w_F': 0.3}
}
```


## 📊 Visualization

The system automatically generates 12+ types of professional charts:

1. **Main Comparison** - Trust Score vs Accuracy
2. **Detailed Metrics Analysis** - All Trust-ADE components
3. **CUDA vs CPU Comparison** - Performance and quality analysis
4. **Correlation Analysis** - Relationships between metrics
5. **Domain-specific Analysis** - Per-dataset breakdowns
6. **Model Performance Heatmaps** - Comprehensive metric overview

### Custom Visualization

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
create_fixed_visualizations(df_viz, 'results', '20250823_custom')
```


## 🧪 Testing

```bash
# Installation check
python tests/test_installation.py

# Basic tests
python tests/test_basic.py

# Test specific module
python -c "from training.trainers import train_models; print('✅ Trainers OK')"
```


## 🔧 System Extension

### Adding New Dataset

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


### New Model Type

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


## 🚀 Usage Examples

### Medical Diagnosis

```bash
python main.py --datasets breast_cancer --verbose
```


### Financial Analysis

```bash
python main.py --datasets wine --cuda-only
```


### Quick Comparison

```bash
python main.py --datasets iris --quick
```


## 📄 Execution Logs

Example detailed log:

```
🔬 ADVANCED ML MODEL COMPARISON WITH TRUST-ADE PROTOCOL + CUDA
================================================================================
📊 TESTING ON DATASET: BREAST_CANCER
📋 Description: Breast cancer diagnosis (2 classes, 30 features)
🏷️ Domain: medical
🎯 Task type: binary
================================================================================

  📈 Training XANFIS...
    ✅ XANFIS trained in 31.28 sec, accuracy: 0.550
    🧠 Rules extracted: 19
    📊 Explainability: Full rule and feature importance support

🔍 Enhanced Trust-ADE model evaluation...
  📊 Evaluating XANFIS...
    🎯 Trust Score: 0.863
    📊 Trust Level: High Trust
    📈 Metrics: Bias=0.253, Drift=0.031
```


## 🔬 Scientific Foundation

The Trust-ADE protocol is based on research:

- **Causal interpretability** instead of correlations
- **Dynamic monitoring** of explanation quality
- **Integrated assessment** of explainability, robustness and fairness
- **Standards compliance** ISO/IEC 24029 and EU AI Act


### Why XANFIS Achieves Higher Trust Scores

1. **🧠 Rule-Based Explainability**: Unlike black-box models, XANFIS provides explicit IF-THEN rules
2. **🔍 Causal Reasoning**: Rules represent causal relationships rather than correlations
3. **📊 Feature Coverage**: Complete feature space coverage with interpretable membership functions
4. **⚖️ Explanation Stability**: Rule-based explanations are inherently more stable than post-hoc methods
5. **🎯 Human Comprehensibility**: Rules can be directly understood by domain experts

### Component Formulas

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


## 📄 Citation

If you use Trust-ADE in your research, please cite:

```bibtex
@article{trofimov2025trust_ade,
  title={From Correlations to Causality: XAI 2.0 Approach and Trust-ADE Protocol for Trustworthy AI},
  author={Trofimov, Yu.V. and Averkin, A.N. and Ilyin, A.S. and Lebedev, A.D.},
  journal={Journal of Explainable AI},
  year={2025}
}
```


## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Create Pull Request

## 📄 License

MIT License - free use for research and commercial projects.

***

**Trust-ADE** — Your reliable tool for creating trustworthy artificial intelligence systems based on a scientifically grounded protocol for dynamic assessment of explainability, robustness and fairness!

