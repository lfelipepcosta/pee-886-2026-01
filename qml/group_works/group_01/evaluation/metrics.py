import matplotlib.pyplot as plt
import os

def plot_comparison(hybrid_history, classical_history, output_dir="../../../data/group_works/group_01/"):
    """
    Plota as métricas de treinamento e validação comparando os modelos Híbrido e Clássico.
    Salva a imagem do gráfico no diretório de saída especificado.
    
    Args:
        hybrid_history (dict): Histórico de métricas do modelo Híbrido.
        classical_history (dict): Histórico de métricas do modelo Clássico.
        output_dir (str): Caminho onde o gráfico será salvo.
    """
    # Define o eixo x (épocas)
    epochs = range(1, len(hybrid_history['train_loss']) + 1)
    
    # Cria a figura e os dois subgráficos (Loss e Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de Perda
    ax1.plot(epochs, hybrid_history['train_loss'], 'b-', label='Perda de Treino (Híbrido)')
    ax1.plot(epochs, hybrid_history['val_loss'], 'b--', label='Perda de Valid. (Híbrido)')
    ax1.plot(epochs, classical_history['train_loss'], 'r-', label='Perda de Treino (Clássico)')
    ax1.plot(epochs, classical_history['val_loss'], 'r--', label='Perda de Valid. (Clássico)')
    ax1.set_title('Perda de Treinamento e Validação')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Perda')
    ax1.set_xticks(epochs) # Garante que o eixo X mostre apenas números inteiros
    ax1.legend() # Mostra a legenda
    ax1.grid(True) # Ativa as grades para facilitar a leitura
    
    # Gráfico de Acurácia
    ax2.plot(epochs, hybrid_history['train_acc'], 'b-', label='Acc. de Treino (Híbrido)')
    ax2.plot(epochs, hybrid_history['val_acc'], 'b--', label='Acc. de Valid. (Híbrido)')
    ax2.plot(epochs, classical_history['train_acc'], 'r-', label='Acc. de Treino (Clássico)')
    ax2.plot(epochs, classical_history['val_acc'], 'r--', label='Acc. de Valid. (Clássico)')
    ax2.set_title('Acurácia de Treinamento e Validação')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Acurácia')
    ax2.set_xticks(epochs) # Garante que o eixo X mostre apenas números inteiros
    ax2.legend() # Mostra a legenda
    ax2.grid(True) # Ativa as grades para facilitar a leitura
    
    # Ajusta o espaçamento
    plt.tight_layout()
    
    # Salva o gráfico como imagem
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "comparacao_modelos.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Gráfico comparativo salvo com sucesso em: {fig_path}")
    
    # Exibe o gráfico no notebook
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """
    Gera e salva a matriz de confusão para o modelo.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
    plt.title(f'Matriz de Confusao - {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name.lower()}.png"), dpi=300)
    plt.show()
