"""
CUDA-оптимизированные модели
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.settings import DEVICE, CUDA_EFFICIENT_THRESHOLD


class OptimizedCUDAMLPClassifier:
    """Оптимизированная PyTorch MLP с адаптивным CUDA использованием"""

    def __init__(self, hidden_layers=(100, 50), n_classes=2, learning_rate=0.001,
                 epochs=300, device='cuda', random_state=42, dataset_size=0):
        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

        # Адаптивный выбор устройства на основе размера данных
        if dataset_size < CUDA_EFFICIENT_THRESHOLD:
            self.device = 'cpu'
            self.use_cuda = False
            print(f"      📱 Используем CPU (датасет мал: {dataset_size} < {CUDA_EFFICIENT_THRESHOLD})")
        else:
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.use_cuda = self.device == 'cuda'
            print(f"      🚀 Используем CUDA (датасет большой: {dataset_size})")

        # Адаптивные параметры в зависимости от размера данных
        if dataset_size < 200:
            self.epochs = 150
            self.batch_size = min(16, dataset_size // 4)
        elif dataset_size < 500:
            self.epochs = 200
            self.batch_size = 32
        else:
            self.epochs = 300
            self.batch_size = 64

        # Устанавливаем seed
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)

    def _create_model(self, input_size):
        """Создание адаптивной PyTorch модели"""
        layers = []

        # Адаптивная архитектура
        if input_size < 10:
            hidden_sizes = [max(8, input_size * 2), max(4, input_size)]
        elif input_size < 50:
            hidden_sizes = self.hidden_layers
        else:
            hidden_sizes = (min(512, input_size * 2), 256, 128)

        # Входной слой
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))

        # Скрытые слои
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        # Выходной слой
        layers.append(nn.Linear(hidden_sizes[-1], self.n_classes))

        return nn.Sequential(*layers)

    def fit(self, X, y):
        """Обучение с адаптивными параметрами"""
        # Нормализация данных
        X_scaled = self.scaler.fit_transform(X)

        # Определяем количество классов
        self.n_classes = len(np.unique(y))

        # Создаем модель
        self.model = self._create_model(X_scaled.shape[1]).to(self.device)

        # Конвертируем в тензоры
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Адаптивный оптимизатор и learning rate
        if X.shape[0] < 200:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate * 2)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        criterion = nn.CrossEntropyLoss()

        # Обучение с адаптивным логированием
        self.model.train()
        log_interval = max(25, self.epochs // 8)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % log_interval == 0:
                print(f"      Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")

        return self

    def predict(self, X):
        """Предсказание классов"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()

