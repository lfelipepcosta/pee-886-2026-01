import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import urllib.request
import zipfile
import shutil

def download_and_prepare_dataset(data_dir):
    """
    Baixa um dataset público de MRI do GitHub se ele não existir localmente
    e organiza as pastas em 'benign' e 'malignant'.
    """
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print(f"Dataset já encontrado em {data_dir}. Pulando o download.")
        return

    print("Baixando o dataset de MRI (SartajBhuvaji/Brain-Tumor-Classification-DataSet)")
    os.makedirs(data_dir, exist_ok=True)
    
    # URL do repositório zipado no GitHub
    url = "https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/archive/refs/heads/master.zip"
    zip_path = os.path.join(data_dir, "dataset.zip")
    
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extraindo arquivos")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    os.remove(zip_path)
    
    # Organiza as pastas: Meningioma = Benigno (geralmente), Glioma = Maligno (geralmente)
    base_extracted_folder = os.path.join(data_dir, "Brain-Tumor-Classification-DataSet-master", "Training")
    
    benign_dir = os.path.join(data_dir, "benign")
    malignant_dir = os.path.join(data_dir, "malignant")
    
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)
    
    # Move as imagens do Meningioma para Benigno
    print("Organizando classes (Benigno vs Maligno)")
    meningioma_path = os.path.join(base_extracted_folder, "meningioma_tumor")
    if os.path.exists(meningioma_path):
        for file in os.listdir(meningioma_path):
            shutil.move(os.path.join(meningioma_path, file), os.path.join(benign_dir, file))
            
    # Move as imagens do Glioma para Maligno
    glioma_path = os.path.join(base_extracted_folder, "glioma_tumor")
    if os.path.exists(glioma_path):
        for file in os.listdir(glioma_path):
            shutil.move(os.path.join(glioma_path, file), os.path.join(malignant_dir, file))
            
    # Limpa as pastas extras que não vamos usar (Pituitary, No Tumor, Testing)
    shutil.rmtree(os.path.join(data_dir, "Brain-Tumor-Classification-DataSet-master"))
    print("Download e preparação concluídos com sucesso!")


def get_dataloaders(data_dir="../../../data/group_works/group_01/mri_dataset", batch_size=32, train_split=0.7, val_split=0.15, download=True):
    """
    Retorna os data loaders para o dataset de MRI, dividindo em treino, validação e teste.
    
    Args:
        data_dir (str): Caminho para o diretório de dados.
        batch_size (int): Tamanho do lote.
        train_split (float): Proporção para treino.
        val_split (float): Proporção para validação.
        download (bool): Se True, tenta baixar o dataset da internet.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if download:
        download_and_prepare_dataset(data_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if os.path.exists(data_dir):
        try:
            full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
            
            if len(full_dataset.classes) < 2:
                raise ValueError("Número insuficiente de classes")
                
            n_total = len(full_dataset)
            n_train = int(train_split * n_total)
            n_val = int(val_split * n_total)
            n_test = n_total - n_train - n_val
            
            train_dataset, val_dataset, test_dataset = random_split(full_dataset, [n_train, n_val, n_test])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"Erro ao carregar os dados reais: {e}")
            
    print("Aviso: Diretório de dados vazio ou falhou. Gerando dados fictícios (dummy) para teste")
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            img = torch.randn(3, 224, 224)
            label = torch.randint(0, 2, (1,)).item()
            return img, label
            
    train_loader = DataLoader(DummyDataset(100), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DummyDataset(20), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(DummyDataset(20), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_kfold_dataloaders(data_dir="../../../data/group_works/group_01/mri_dataset", batch_size=32, n_splits=5, download=True):
    """
    Retorna uma lista de tuplas (train_loader, val_loader) para K-Fold.
    """
    if download:
        download_and_prepare_dataset(data_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(data_dir):
        return []

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    from torch.utils.data import Subset
    from sklearn.model_selection import KFold
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    
    for train_ids, val_ids in kfold.split(full_dataset):
        train_sub = Subset(full_dataset, train_ids)
        val_sub = Subset(full_dataset, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
        folds.append((train_loader, val_loader))
        
    return folds
