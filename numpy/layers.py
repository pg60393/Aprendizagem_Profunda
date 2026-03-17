import numpy as np

class Layer:
    def forward(self, inputs, training=True):
        pass
    def backward(self, dvalues):
        pass

class Dense(Layer):
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l2=0.0):
        # Inicialização de pesos (He initialization para melhor convergência)
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l2 = weight_regularizer_l2

    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradientes da regularização L2
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs
            return self.output
            
        # Máscara binomial escalada (Inverted Dropout)
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
        self.output = inputs * self.mask
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * self.mask
        return self.dinputs