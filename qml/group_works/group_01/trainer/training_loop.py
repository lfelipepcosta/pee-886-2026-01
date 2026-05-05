import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os

def train_model(model, train_loader, val_loader, model_name="Model", epochs=5, lr=0.0004, weight_decay=1e-4, device="cpu", output_dir="../../../data/group_works/group_01/", verbose=True):
    """
    Treina o modelo fornecido e retorna o histórico de métricas (treino e validação).
    Salva também um relatório em formato txt no diretório de saída especificado.
    
    Args:
        model (torch.nn.Module): Modelo PyTorch a ser treinado.
        train_loader (DataLoader): DataLoader para os dados de treinamento.
        val_loader (DataLoader): DataLoader para os dados de validação.
        model_name (str): Nome do modelo para o arquivo de saída.
        epochs (int): Número de épocas de treinamento.
        lr (float): Taxa de aprendizado (learning rate).
        device (torch.device ou str): Dispositivo de execução ('cpu' ou 'cuda').
        output_dir (str): Caminho para salvar os resultados.
        
    Returns:
        dict: Dicionário contendo o histórico de perda e acurácia.
    """
    model = model.to(device)
    
    # Define a função de perda (CrossEntropy para classificação multiclasse/binária)
    criterion = nn.CrossEntropyLoss()
    
    # Define o otimizador Adam com a taxa de aprendizado e decaimento de pesos (L2 penalty) para evitar overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Se a acurácia de validação parar de subir, reduzimos o passo (lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # Guarda a melhor acurácia para salvar o checkpoint
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'time': 0}
    start_time = time.time()
    
    for epoch in range(epochs):
        # model.train(): Coloca o modelo em modo de treinamento.
        # Ativas camadas como Dropout e BatchNorm para se ativarem e 
        # aprenderem com os dados. Sem isso, o modelo não treina corretamente.
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        # Fase de Treinamento
        iterator = tqdm(train_loader, desc=f"Treinando [{model_name}]", disable=not verbose)
        for inputs, labels in iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zera os gradientes acumulados
            optimizer.zero_grad()
            
            # Passo Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Passo Backward e Otimização
            loss.backward()
            optimizer.step()
            
            # Acumula a perda e as predições corretas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        # Calcula as métricas médias da época para treino
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # Fase de Validação
        # model.eval(): Coloca o modelo em modo de avaliação/inferência
        # Isso desliga o Dropout e o BatchNorm (usa os pesos aprendidos no treino)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        # torch.no_grad(): Desliga o cálculo matemático de gradientes (derivadas)
        # Como na validação não estamos aprendendo nada novo, desligar os gradientes
        # economiza muita memória da GPU e acelera o processo consideravelmente
        with torch.no_grad():
            iterator_val = tqdm(val_loader, desc="Validando", leave=False, disable=not verbose)
            for inputs, labels in iterator_val:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # Calcula as métricas médias da época para validação
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        if verbose:
            print(f"Época {epoch+1}/{epochs} | Treino Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_val_acc:.4f}")
        
        # Atualiza Scheduler
        scheduler.step(epoch_val_acc)
        
        # Salva o melhor modelo
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f"{model_name.replace(' ', '')}_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f"Novo melhor modelo salvo: {best_val_acc:.4f}")
        
    history['time'] = time.time() - start_time
    
    # Salva as métricas calculadas em um arquivo txt
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name.replace(' ', '')}_metrics.txt"
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(f"Relatório {model_name}\nTempo: {history['time']:.2f}s\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
            
    return history

def test_model(model, test_loader, device="cpu"):
    """
    Executa a inferência final em um conjunto de teste
    """
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    
    print(f"Iniciando inferência final (Blind Test)")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inferencia Final"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = correct / total
    print(f"Acuracia no teste cego: {accuracy:.4f}")
    return accuracy, all_preds, all_labels
