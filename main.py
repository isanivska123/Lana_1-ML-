import numpy as np
import matplotlib.pyplot as plt

# Функція для визначення середньої квадратичної помилки
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Клас для лінійної регресії
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Обчислення градієнтів
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Оновлення ваг та зміщення
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Виведення значень ваг та функції втрат
            print(f'Epoch {_ + 1}: weights = {self.weights}, bias = {self.bias}, loss = {mean_squared_error(y, y_pred)}')
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Зчитування даних з train датасету
data_train = np.loadtxt('lab1_train.csv', delimiter=',', skiprows=1)
X_train = data_train[:, 1].reshape(-1, 1)  # Ознаки
y_train = data_train[:, 2]  # Мітки

# Навчання моделі лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)

# Зчитування даних з test датасету
data_test = np.loadtxt('lab1_test.csv', delimiter=',', skiprows=1)
X_test = data_test[:, 1].reshape(-1, 1)  # Ознаки
y_test = data_test[:, 2]  # Мітки

# Передбачення на тестових даних
y_pred_test = model.predict(X_test)
test_loss = mean_squared_error(y_test, y_pred_test)
print(f'Test loss: {test_loss}')

# Візуалізація даних з lab_1_train.csv
plt.scatter(X_train, y_train, color='blue', label='Train data')
# Візуалізація даних з lab1__test.csv
plt.scatter(X_test, y_test, color='red', label='Test data')
# Візуалізація лінії, до якої зійшовся розв'язок на train даних
plt.plot(X_train, model.predict(X_train), color='green', label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

plt....
