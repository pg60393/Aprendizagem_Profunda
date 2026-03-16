import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Cortar valores para evitar log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # Labels categóricas (ex: [0, 1, 4])
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            # Labels em One-hot encoding
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)

    def backward(self, y_pred, y_true):
        samples = len(y_pred)
        labels = len(y_pred[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Gradiente da loss
        self.dinputs = -y_true / y_pred
        self.dinputs = self.dinputs / samples
        return self.dinputs