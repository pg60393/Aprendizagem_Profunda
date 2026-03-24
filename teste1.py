import numpy as np
import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def treinar_melhor_modelo():
    print("A carregar dados...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    print("A treinar LinearSVC (O Campeão)...")
    # Softer weights to prevent the "Garbage Bin" effect
    custom_weights = {
        0: 0.8,   # Human
        1: 4.0,   # OpenAI (Still a boost, but not 25x!)
        2: 1.0,   # Google
        3: 0.8,   # Meta
        4: 1.5    # Anthropic
    }
    
    # Go back to the standard C=0.5 so it doesn't overthink the fragments
    model = LinearSVC(
        C=0.5, 
        class_weight=custom_weights, 
        max_iter=3000, 
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save the model
    os.makedirs('Subm1', exist_ok=True)
    with open('Subm1/melhor_modelo_svc.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Test it
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Precisão Final do LinearSVC no TEST SET: {acc * 100:.2f}%")

if __name__ == "__main__":
    treinar_melhor_modelo()