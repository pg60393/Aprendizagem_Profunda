import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import os

def set_seed(seed=42):
    #Definir a seed para garantir reprodutibilidade
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def carregar_dados_torch():
    print("A carregar dados para PyTorch...")
    X_train = np.load('X_train.npy').astype(np.float32)
    X_val = np.load('X_val.npy').astype(np.float32)
    X_test = np.load('X_test.npy').astype(np.float32)
    
    y_train = np.load('y_train.npy').astype(np.int64) 
    y_val = np.load('y_val.npy').astype(np.int64)
    y_test = np.load('y_test.npy').astype(np.int64)
    return X_train, y_train, X_val, y_val, X_test, y_test

class TextoClassificadorDNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextoClassificadorDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.5), 
            
            nn.Linear(input_size, 32),
            nn.ReLU(),
            
            nn.Dropout(0.5),
            
            nn.Linear(32, num_classes) 
        )

    def forward(self, x):
        return self.net(x)

def treinar_modelo():
    SEED = 42
    set_seed(SEED)
    
    X_train, y_train, X_val, y_val, X_test, y_test = carregar_dados_torch()
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    input_size = X_train.shape[1]
    num_classes = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar no dispositivo: {device}")
    
    model = TextoClassificadorDNN(input_size, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 20
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train() 
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()        
            outputs = model(inputs)      
            loss = criterion(outputs, labels) 
            loss.backward()              
            optimizer.step()             
            train_loss += loss.item()
            
        model.eval() 
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('Subm2', exist_ok=True)
            torch.save(model.state_dict(), 'Subm2/melhor_modelo_pytorch.pth')

    print(f"\nTreino concluído! Melhor precisão na validação: {best_val_acc:.2f}%")
    
    model.load_state_dict(torch.load('Subm2/melhor_modelo_pytorch.pth'))
    model.eval()
    
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Precisão Final no Test Set (PyTorch): {100 * correct / total:.2f}%")

if __name__ == "__main__":
    treinar_modelo()