"""
Настройки и константы проекта
"""
import os
import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings('ignore')

# Настройка matplotlib для исправления ошибок форматирования
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Добавляем путь к проекту
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# CUDA проверка с оптимизированным порогом
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
CUDA_EFFICIENT_THRESHOLD = 500  # Используем CUDA только если данных больше 500

if CUDA_AVAILABLE:
    print(f"✅ CUDA доступно: {torch.cuda.get_device_name(0)}")
    print(f"   Устройство: {DEVICE}")
    print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"   ⚠️ CUDA будет использоваться только для датасетов > {CUDA_EFFICIENT_THRESHOLD} образцов")
else:
    print("⚠️ CUDA недоступно, используется CPU")

# Импорт XANFIS с обработкой ошибок
try:
    from xanfis import Data, GdAnfisClassifier, AnfisClassifier
    XANFIS_AVAILABLE = True
    print("✅ XANFIS успешно импортирован")
except ImportError:
    XANFIS_AVAILABLE = False
    print("❌ XANFIS недоступен. Установите: pip install xanfis torch mealpy permetrics")

