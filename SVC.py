import numpy as np
import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def treinar_melhor_modelo():
    print("A carregar dados...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_val = np.load('X_val.npy')
    y_val = np.load('y_val.npy')
    
    X_train_full = np.load('X_train_full.npy')
    y_train_full = np.load('y_train_full.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    custom_weights = {
        0: 0.8,   # Human
        1: 5.0,   # OpenAI
        2: 1.0,   # Google
        3: 0.5,   # Meta
        4: 1.0    # Anthropic
    }

    print("A afinar LinearSVC na Validação...")
    best_c = 0.5
    best_c = 0.1
    best_acc = 0
    
    # Test MUCH lower C values to force generalization
    for c_val in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        model = LinearSVC(C=c_val, class_weight=custom_weights, max_iter=3000, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"  C={c_val} -> Val Acc: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_c = c_val

    print(f"\nMelhor hiperparâmetro encontrado: C={best_c}")
    print("A treinar modelo final em TODO o conjunto de treino (90%)...")
    
    final_model = LinearSVC(C=best_c, class_weight=custom_weights, max_iter=3000, random_state=42)
    final_model.fit(X_train_full, y_train_full)
    
    final_preds = final_model.predict(X_test)
    
    print("\n--- RESULTADOS FINAIS (TEST SET) ---")
    print(f"Precisão: {accuracy_score(y_test, final_preds) * 100:.2f}%\n")
    
    # Remap labels
    target_names = ['Human', 'OpenAI', 'Google', 'Meta', 'Anthropic']
    print(classification_report(y_test, final_preds, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, final_preds))

    os.makedirs('Subm2', exist_ok=True)
    with open('Subm2/melhor_modelo_svc.pkl', 'wb') as f:
        pickle.dump(final_model, f)

if __name__ == "__main__":
    treinar_melhor_modelo()