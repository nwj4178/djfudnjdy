import numpy as np
import nnfs
from nnfs.datasets import spiral_data  # 10개의 클래스를 위한 데이터셋

nnfs.init()

# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, inputs):
        self.dweights = np.dot(inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# Relu Activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues, inputs):
        self.dinputs = dvalues.copy()
        self.dinputs[inputs <= 0] = 0


# Softmax Activation
class Activation_Softmax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)


# 손실 함수
class Loss_CategoricalCrossentropy:
    def calculate(self, output, y):
        samples = len(output)
        y_pred_clipped = np.clip(output, 1e-7, 1 - 1e-7)

        if len(y.shape) == 1:  # 정수형 라벨
            correct_confidences = y_pred_clipped[range(samples), y]
        else:
            correct_confidences = np.sum(y_pred_clipped * y, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, dvalues, y):
        samples = len(dvalues)
        dvalues = dvalues.copy()

        if len(y.shape) == 1:
            dvalues[range(samples), y] -= 1
        else:
            dvalues -= y

        return dvalues / samples


# 데이터셋 준비 (클래스 10개로)
X, y = spiral_data(samples=1000, classes=10)

# 첫 레이어 (2개 입력, 64개 뉴런)
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()

# 두 번째 레이어 (64개 입력, 10개 출력)
dense2 = Layer_Dense(64, 10)
activation2 = Activation_Softmax()

# 손실 함수 설정
loss_function = Loss_CategoricalCrossentropy()

# 학습률 설정
learning_rate = 0.01

# 학습 과정 (반복 횟수 20,000번)
for iteration in range(20000):
    # 순전파
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # 손실 계산
    loss = loss_function.calculate(activation2.output, y)

    # 정확도 계산
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # 출력 (10번에 한 번씩)
    if iteration % 100 == 0:
        print(f"iteration: {iteration}, loss: {loss:.4f}, acc: {accuracy:.4f}")

    # 역전파 (백프로파게이션)
    dvalues = loss_function.backward(activation2.output, y)
    dense2.backward(dvalues, activation1.output)
    activation1.backward(dense2.dinputs, dense1.output)
    dense1.backward(activation1.dinputs, X)

    # 가중치와 편향 업데이트
    dense1.weights -= learning_rate * dense1.dweights
    dense1.biases -= learning_rate * dense1.dbiases
    dense2.weights -= learning_rate * dense2.dweights
    dense2.biases -= learning_rate * dense2.dbiases