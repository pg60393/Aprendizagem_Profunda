"""import numpy as np

class Activation:
    def forward(self, inputs):
        pass
    def backward(self, dvalues):
        pass

class ReLU(Activation):
    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Softmax(Activation):
    def forward(self, inputs, training=True):
        self.inputs = inputs
        # Subtrair o max para estabilidade numérica (evitar overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        # O gradiente da softmax é calculado iterando sobre as amostras
        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs"""

import numpy as np

class Activation:
    def forward(self, inputs):
        pass
    def backward(self, dvalues):
        pass

class ReLU(Activation):
    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        # ATENÇÃO: A ReLU tem de manter esta regra (meter a 0 os gradientes dos inputs negativos)
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs

class Softmax(Activation):
    def forward(self, inputs, training=True):
        self.inputs = inputs
        # Subtrair o max para estabilidade numérica (evitar overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        return self.dinputs