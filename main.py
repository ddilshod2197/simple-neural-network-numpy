import numpy as np

class NeyronTarmoq:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Yashirin qavatning ogohlantiruvchi koeffitsientlari
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))

        # Chiqish qavatining ogohlantiruvchi koeffitsientlari
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        # Yashirin qavatga kirish
        self.hidden_layer = self.sigmoid(np.dot(inputs, self.weights1) + self.bias1)

        # Chiqish qavatiga kirish
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)

        return self.output_layer

    def backpropagation(self, inputs, targets, learning_rate):
        # Chiqish qavatining hata darajasi
        output_error = targets - self.output_layer
        output_delta = output_error * self.sigmoid_derivative(self.output_layer)

        # Yashirin qavatining hata darajasi
        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer)

        # Ogohlantiruvchi koeffitsientlarni yangilash
        self.weights2 += learning_rate * self.hidden_layer.T.dot(output_delta)
        self.bias2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.weights1 += learning_rate * inputs.T.dot(hidden_delta)
        self.bias1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            self.forward(inputs)
            self.backpropagation(inputs, targets, learning_rate)

# Misol uchun ma'lumotlar
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Neyron tarmoqni yaratish
neuron = NeyronTarmoq(2, 1, 2)

# Neyron tarmoqni mashq qilish
neuron.train(inputs, targets, 0.1, 10000)

# Chiqish qavatining natijalarini chiqarish
print(neuron.output_layer)
