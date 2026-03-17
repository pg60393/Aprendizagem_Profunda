import numpy as np
import pickle
import os
from layers import Dense, Dropout
from activation import ReLU, Softmax
from losses import CategoricalCrossEntropy
from optimizer import SGD
from neuralnet import NeuralNetwork
from metrics import accuracy

def carregar_dados():
    print("A carregar dados tabulares...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    return X_train, y_train, X_val, y_val

def treinar_baseline(X_train, y_train, X_val, y_val):
    print("\n--- A Treinar Baseline: Regressão Logística (Numpy) ---")
    # Regressão Logística é apenas uma camada Densa ligada diretamente a uma Softmax
    num_features = X_train.shape[1]
    num_classes = 5
    
    model = NeuralNetwork()
    model.add(Dense(num_features, num_classes))
    model.add(Softmax())
    
    model.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(learning_rate=0.1, decay=1e-4))
    model.fit(X_train, y_train, epochs=100, X_val=X_val, y_val=y_val, patience=5)
    return model

def treinar_dnn(X_train, y_train, X_val, y_val):
    print("\n--- A Treinar Deep Neural Network (DNN com Dropout e Regularização) ---")
    num_features = X_train.shape[1]
    num_classes = 5
    
    model = NeuralNetwork()
    # Camada Oculta 1 com Regularização L2
    model.add(Dense(num_features, 128, weight_regularizer_l2=5e-4))
    model.add(ReLU())
    model.add(Dropout(0.2)) 
    
    # Camada Oculta 2
    model.add(Dense(128, 64))
    model.add(ReLU())
    model.add(Dropout(0.2))
    
    # Camada de Saída
    model.add(Dense(64, num_classes))
    model.add(Softmax())
    
    # Melhoria: SGD com Momentum
    model.compile(loss=CategoricalCrossEntropy(), optimizer=SGD(learning_rate=0.05, decay=1e-4, momentum=0.9))
    model.fit(X_train, y_train, epochs=200, X_val=X_val, y_val=y_val, patience=15)
    return model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = carregar_dados()
    
    modelo_base = treinar_baseline(X_train, y_train, X_val, y_val)

    preds_base = modelo_base.forward(X_val, training=False)
    acc_base = accuracy(preds_base, y_val)
    print(f"\nPrecisão Final da Baseline (Regressão Logística): {acc_base * 100:.2f}%")
    
    modelo_dnn = treinar_dnn(X_train, y_train, X_val, y_val)

    os.makedirs('Subm1', exist_ok=True)
    with open('Subm1/modelo_numpy.pkl', 'wb') as f:
        pickle.dump(modelo_dnn, f)    
    
    # Teste final nos dados de validação
    preds_dnn = modelo_dnn.forward(X_val, training=False)
    acc_final = accuracy(preds_dnn, y_val)
    print(f"\nPrecisão Final do Modelo DNN na Validação: {acc_final * 100:.2f}%")