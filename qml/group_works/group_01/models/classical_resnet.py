import torch
import torch.nn as nn
import torchvision.models as models

class ClassicalResNet18(nn.Module):
    """
    Modelo clássico de classificação de imagens para comparação (Baseline).
    Utiliza a ResNet-18 para extração de características.
    """
    def __init__(self, num_classes=2):
        """
        Inicializa a arquitetura clássica.
        
        Args:
            num_classes (int): Número de classes alvo (2 para benigno/maligno)
        """
        super().__init__()
        # Carrega a rede ResNet-18 pré-treinada (ImageNet)
        self.resnet = models.resnet18(pretrained=True)
        
        # Congela todas as camadas convolucionais (Backbone)
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Pega o número de atributos da última camada totalmente conectada (512 atributos)
        num_ftrs = self.resnet.fc.in_features  
        
        # Para que a comparação seja justa em número de parâmetros simulados na redução,
        # adicionamos uma camada escondida que reduz a 4 atributos (mesmo gargalo do modelo híbrido)
        # antes da classificação final
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4),     # Reduz aos 4 atributos
            nn.ReLU(),                  # Ativação não-linear
            nn.Linear(4, num_classes)   # Classificação nas 2 classes
        )

    def forward(self, x):
        """
        Passo forward do modelo clássico
        """
        return self.resnet(x)
