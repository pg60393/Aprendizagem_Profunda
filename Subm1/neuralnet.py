import numpy as np
from metrics import accuracy

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss_function = loss
        self.optimizer = optimizer

    def forward(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training)
        return output

    def backward(self, output, y):
        # O backward começa pela loss function
        dinputs = self.loss_function.backward(output, y)
        # E propaga de trás para a frente pelas camadas
        for layer in reversed(self.layers):
            dinputs = layer.backward(dinputs)

    def fit(self, X_train, y_train, epochs, X_val=None, y_val=None, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Treino
            output = self.forward(X_train, training=True)
            loss = self.loss_function.calculate(output, y_train)
            acc = accuracy(output, y_train)

            self.backward(output, y_train)

            self.optimizer.pre_update_params()
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Validação e Early Stopping
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val, training=False)
                val_loss = self.loss_function.calculate(val_output, y_val)
                val_acc = accuracy(val_output, y_val)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch} | Train Loss: {loss:.4f}, Acc: {acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                # Early Stopping Logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0 # Reset
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                    break
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch} | Train Loss: {loss:.4f}, Acc: {acc:.4f}")