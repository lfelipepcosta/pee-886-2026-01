# Documentacao Completa do Pipeline: Classificador Hibrido Quantico para MRI

**Projeto:** PEE-886 - Aprendizado de Maquina Quantico  
**Grupo:** Group 01  
**Tarefa:** Classificacao Binaria de Tumores Cerebrais (Benigno vs Maligno)

---

## Visao Geral

O pipeline implementado segue o padrao cientifico de Machine Learning, composto por quatro fases sequenciais:

```
[1] Otimizacao de Hiperparametros (Optuna)
         |
         v
[2] Validacao Cruzada (K-Fold)
         |
         v
[3] Treinamento Final
         |
         v
[4] Inferencia em Teste Cego (Blind Test)
```

Cada fase foi projetada para garantir que os resultados sejam **reprodutiveis**, **generalizaveis** e **confiavéis** para contexto de pesquisa.

---

## Fase 0: O Dataset

### Origem e Composicao
O dataset utilizado e o **SartajBhuvaji/Brain-Tumor-Classification**, disponivel publicamente no GitHub. Ele contem imagens de ressonancia magnetica (MRI) cerebral divididas em quatro categorias originais: Glioma, Meningioma, Pituitario e Sem Tumor.

### Reclassificacao Binaria
Para simplificar o problema e focar na utilidade clinica (triagem inicial), reclassificamos as imagens em apenas duas classes:
- **Benigno**: Tumores do tipo **Meningioma**.
- **Maligno**: Tumores do tipo **Glioma**.

**O que acontece com o Pituitario e o "No Tumor"?** Essas duas categorias do dataset original sao **completamente descartadas** pelo loader. A razao e que o Pituitario e um tipo ambiguo (pode ser benigno ou maligno dependendo do subtipo) e inclui-lo forçaria uma classificacao binaria artificialmente imprecisa. O "No Tumor" nao e relevante para o problema de triagem que estamos resolvendo. Assim, o problema fica limpo: **Meningioma = Benigno, Glioma = Maligno**.

### Pre-processamento das Imagens
Cada imagem passa por duas transformacoes antes de entrar no modelo:
1. **Resize para 224x224 pixels:** A ResNet-18 foi projetada para imagens nesse tamanho. Usar outra resolucao exigiria modificar a arquitetura.
2. **Normalizacao com estatisticas do ImageNet:** Os valores dos pixels de uma imagem vao de 0 a 255. Apos `ToTensor()`, eles vao de 0.0 a 1.0. Porem, os pesos pre-treinados da ResNet foram calculados assumindo que as imagens teriam media e variancia especificas (media `[0.485, 0.456, 0.406]` e desvio padrao `[0.229, 0.224, 0.225]` para os canais R, G e B, respectivamente). Esses valores foram calculados sobre o dataset ImageNet inteiro.
   - **O que a normalizacao faz?** Ela reescala cada pixel para que a distribuicao dos valores tenha media zero e desvio padrao um (aproximadamente). Isso garante que os dados de entrada "falem a mesma lingua" que os pesos da ResNet esperavam encontrar.
   - **O que aconteceria sem ela?** As primeiras camadas convolucionais da ResNet receberiam valores em um range diferente do esperado. Isso causaria gradientes explosivos ou nulos no inicio do treinamento, tornando o aprendizado instavel ou impossivel.

### Divisao dos Dados (70/15/15)
Os dados sao divididos de forma estratificada em tres conjuntos:
- **Treino (70%):** Usado para ajustar os pesos do modelo a cada epoca.
- **Validacao (15%):** Usado para monitorar overfitting e decidir quando salvar o checkpoint. O modelo **nunca aprende** com esses dados diretamente.
- **Teste Cego (15%):** Completamente isolado. E usado uma unica vez, ao final de tudo, para medir a performance real do modelo em dados ineditos. Isso simula o uso clinico real.

**Por que 70/15/15 e nao 80/20?** A divisao simetrica entre validacao e teste garante que tenhamos dados suficientes para ambas as funcoes. Um conjunto de teste muito pequeno tornaria a metrica de acuracia final estatisticamente instavel.

---

## Fase 1: Otimizacao de Hiperparametros (Optuna)

**Script:** `scripts/group_works/group_01/run_optimization.py`

### O que sao Hiperparametros?
Hiperparametros sao configuracoes do modelo que **nao sao aprendidas** durante o treinamento. O modelo aprende os *pesos* (parametros), mas o *Learning Rate* ou a *profundidade do circuito quantico* precisam ser escolhidos por nos antes.

Uma escolha ruim de hiperparametros pode fazer um modelo excelente parecer fraco. Por isso, automatizamos essa busca.

### Por que Optuna?
O Optuna e uma biblioteca de otimizacao automatica de hiperparametros que usa o algoritmo **TPE (Tree-structured Parzen Estimator)**. Diferente de uma busca em grade (grid search) que testa todas as combinacoes, o TPE aprende com os trials anteriores para sugerir combinacoes mais promissoras. Isso torna a busca muito mais eficiente.

### O que foi otimizado?
Para o **modelo classico** (ClassicalResNet18):
- `lr` (Learning Rate): Taxa de aprendizado do otimizador. Busca entre `1e-5` e `1e-2` em escala logaritmica. Valores muito altos causam instabilidade; muito baixos, convergencia lenta.
- `weight_decay`: Penalidade L2 nos pesos (regularizacao). Busca entre `1e-6` e `1e-2`. Evita que pesos crescam demais e causem overfitting.

Para o **modelo hibrido** (HybridResNet18):
- Os mesmos `lr` e `weight_decay`.
- `q_depth`: Profundidade do circuito quantico (numero de camadas `StronglyEntanglingLayers`). Busca entre 1 e 3. Mais camadas aumentam a expressividade do circuito mas tambem aumentam o risco de overfitting e o tempo de simulacao.

### Como funciona cada Trial?
Cada "trial" do Optuna e uma execucao completa de treinamento (5 epocas) com uma combinacao de hiperparametros sugerida pelo algoritmo TPE. Ao final, retorna-se a melhor acuracia de validacao atingida naquelas 5 epocas.

**Por que apenas 5 epocas por trial?** O objetivo do Optuna nao e treinar o modelo definitivo, mas apenas estimar se aquela combinacao de parametros e promissora. 5 epocas sao suficientes para ver se o modelo esta aprendendo. O treinamento definitivo acontece na Fase 3.

### Salvamento dos Resultados
- `best_params_classical.json`: Os melhores hiperparametros encontrados para o classico.
- `best_params_hybrid.json`: Os melhores hiperparametros encontrados para o hibrido.
- `optuna/best_classical_checkpoint.pth` e `best_hybrid_checkpoint.pth`: O checkpoint do melhor modelo de cada tipo encontrado durante os trials. Estes arquivos sao **sobrescritos** sempre que um trial melhor e encontrado (nao ha acumulo de arquivos).
- `optimization_time.txt`: Tempo total da fase de otimizacao.

---

## Fase 2: Validacao Cruzada (K-Fold)

**Script:** `scripts/group_works/group_01/run_kfold.py`

### Por que K-Fold e nao apenas validacao simples?
A divisao 70/15/15 pode ser "azarada": dependendo de quais imagens foram parar em qual conjunto, o modelo pode parecer melhor ou pior do que realmente e.

O K-Fold resolve isso dividindo os dados em **K grupos (folds)** e treinando o modelo K vezes, cada vez usando um grupo diferente como validacao. No final, a acuracia reportada e a **media de K rodadas**, o que e muito mais confiavel estatisticamente.

### Configuracao escolhida
- **K=5 Folds**: Padrao da literatura. Oferece bom equilibrio entre custo computacional e confianca estatistica.
- **5 epocas por fold**: Novamente, o objetivo e estimar a estabilidade, nao o maximo desempenho.

### O que o K-Fold garante?
Se a acuracia media foi de 88% com desvio padrao de 2%, isso significa que **independente de como os dados foram divididos**, o modelo consistentemente classifica bem. Isso e muito mais forte do que reportar uma unica acuracia de 88% que pode ter sido "sorte" da divisao.

### Salvamento dos Resultados
- `kfold/kfold_report.txt`: Relatorio unico consolidado com acuracia de cada fold, media e desvio padrao, e tempo medio por fold para ambos os modelos.

---

## Fase 3: Treinamento Final

**Funcao:** `train_model()` em `qml/group_works/group_01/trainer/training_loop.py`

### Diferenca em relacao ao K-Fold
O treinamento final usa **todos os dados de treino** (70%) e valida nos dados de validacao (15%), com os melhores hiperparametros encontrados pelo Optuna. Aqui sao usadas **20 epocas** para garantir convergencia plena.

### Componentes do Loop de Treinamento

**Otimizador Adam:**
Combina dois mecanismos: momentum (memoria da direcao dos gradientes anteriores) e taxas de aprendizado adaptativas por parametro. E o otimizador mais robusto e amplamente usado em Deep Learning.

**Funcao de Perda CrossEntropyLoss:**
Calcula o quanto as probabilidades previstas pelo modelo divergem das classes reais. Para classificacao multiclasse/binaria, e a escolha padrao e matematicamente correta.

**Scheduler ReduceLROnPlateau:**
Se a acuracia de validacao nao melhorar por 2 epocas consecutivas (patience=2), o learning rate e reduzido a metade (factor=0.5). Isso permite que o modelo faca ajustes mais finos quando esta proximo do otimo.

**Checkpointing (Ponto de Recuperacao):**
Imagine que o modelo esta progredindo bem: na epoca 12, ele atinge 88% de acuracia de validacao. Na epoca 13, ele comeca a "decorar" os dados de treino (overfitting) e a acuracia de validacao cai para 85%. Sem checkpointing, voce terminaria com o modelo da epoca 20, que pode ser ainda pior.

Com checkpointing, sempre que a acuracia de validacao supera o maximo historico, o estado completo dos pesos e salvo em um arquivo `.pth`. Se o modelo piorar nas epocas seguintes, o arquivo `.pth` ainda contem os pesos do melhor momento. Ao final do treinamento, usamos **exatamente esses pesos** para a inferencia final, nao os da ultima epoca. Em outras palavras: estamos garantindo que o modelo que chega ao Teste Cego e o modelo no seu melhor momento de generalizacao.

**model.train() e model.eval():**
- `model.train()`: Ativa layers como Dropout e BatchNorm no modo de aprendizado.
- `model.eval()`: As desativa para inferencia, garantindo resultados deterministicos e usando os parametros aprendidos.
- `torch.no_grad()`: Na validacao, desliga o calculo de gradientes para economizar memoria e acelerar o processo.

---

## Fase 4: Inferencia em Teste Cego (Blind Test)

**Funcao:** `test_model()` em `qml/group_works/group_01/trainer/training_loop.py`

### O que e o Teste Cego?
E a unica avaliacao que realmente conta. O conjunto de teste (15% dos dados) foi completamente isolado desde o inicio e nunca foi usado para nenhuma decisao de treinamento ou configuracao. Quando apresentamos o modelo ao conjunto de teste, simulamos exatamente como ele performaria em um hospital recebendo exames novos de pacientes desconhecidos.

**Por que isso e importante?** Se usassemos os dados de validacao para reportar a acuracia final, estariamos sendo desonestos: o modelo foi indiretamente influenciado por esses dados (o checkpoint foi salvo com base neles). O conjunto de teste cego garante zero contaminacao.

### Saidas da Funcao
- `accuracy`: Proporcao de acertos sobre o total de imagens do teste.
- `all_preds`: Lista com a predicao do modelo para cada imagem.
- `all_labels`: Lista com o rotulo real (gabarito) de cada imagem.

As listas de predicoes e rotulos sao usadas para gerar as Matrizes de Confusao.

---

## Arquitetura dos Modelos em Detalhe

### ClassicalResNet18

A ResNet-18 (Residual Network de 18 camadas) foi publicada pela Microsoft Research em 2015. Ela resolveu o problema do **Vanishing Gradient** (gradiente que desaparece em redes muito profundas) com uma ideia elegante: **conexoes residuais (skip connections)**. Em vez de cada camada aprender uma transformacao completamente nova, ela aprende apenas a **diferenca (residuo)** em relacao a entrada. Isso permite treinar redes muito mais profundas sem instabilidade.

A arquitetura interna tem:
- **1 camada convolucional inicial** (7x7, 64 filtros) + MaxPooling.
- **4 blocos residuais** (cada um com 2 camadas conv), extraindo features em resolucoes progressivamente menores mas com mais canais: 64 -> 128 -> 256 -> 512.
- **Global Average Pooling**: Comprime a feature map 7x7 de cada um dos 512 canais em um unico numero, resultando em um vetor de 512 valores.
- **Camada Linear final** (originalmente 512 -> 1000 para o ImageNet).

**O que e o pre-treinamento no ImageNet?** O ImageNet e um dataset com 1.2 milhao de imagens e 1000 categorias. A ResNet foi treinada nesse dataset por semanas em clusters de GPUs. Nesse processo, as camadas convolucionais aprenderam filtros que detectam:
- **Camadas iniciais**: bordas, gradientes de cor, texturas simples.
- **Camadas intermediarias**: partes de objetos (rodas, olhos, texturas complexas).
- **Camadas finais**: conceitos de alto nivel (formas de objetos inteiros).

Essa hierarquia de features e **generalizavel**: os filtros que detectam bordas e texturas em fotos de gatos tambem funcionam para detectar bordas e texturas em imagens de MRI. Por isso o Transfer Learning funciona mesmo entre dominios diferentes.

**O que fazemos com o backbone?**
1. **Congelamos todas as camadas convolucionais** (`requires_grad = False`): Os pesos nao sao atualizados durante o treinamento. Isso economiza tempo e memoria e evita overfitting com poucos dados de MRI.
2. **Substituimos a ultima camada** (512 -> 1000) por nossa cabeca de classificacao.
3. **Apenas a cabeca de classificacao e treinada**: Os pesos das 18 camadas convolucionais ficam fixos; apenas os ~30 parametros da cabeca aprendem a mapear as 512 features para Benigno ou Maligno.

**Por que o gargalo de 4 features?** Para fazer uma comparacao justa com o modelo hibrido, que so pode processar 4 qubits. Assim, ambos os modelos recebem a mesma quantidade de informacao antes de fazer a classificacao.

```
Imagem (224x224x3)
       |
       v
[Camada Conv 7x7 - CONGELADA]  -> detecta bordas e gradientes
[Bloco Residual 1 - CONGELADO] -> detecta texturas (64 filtros)
[Bloco Residual 2 - CONGELADO] -> detecta padroes (128 filtros)
[Bloco Residual 3 - CONGELADO] -> detecta formas complexas (256 filtros)
[Bloco Residual 4 - CONGELADO] -> detecta conceitos de alto nivel (512 filtros)
[Global Average Pooling]        -> 512 valores
       |
       v  (TREINAVEL)
  Linear(512 -> 4)  +  ReLU
  Linear(4 -> 2)
       |
       v
  Logits [Benigno, Maligno]
```

**Por que congelar o backbone?** Com poucos dados de MRI (centenas de imagens), treinar 11 milhoes de parametros do zero causaria overfitting severo. Ao congelar o backbone, aproveitamos o conhecimento visual ja adquirido com milhoes de imagens do ImageNet e treinamos apenas as camadas finais.

**Por que o gargalo de 4 features?** Para fazer uma comparacao justa com o modelo hibrido, que so pode processar 4 qubits. Assim, ambos os modelos recebem a mesma quantidade de informacao da ResNet antes de fazer a classificacao.

### HybridResNet18

```
Imagem (224x224x3)
       |
       v
[ResNet-18 Backbone - CONGELADO]
  (18 camadas convolucionais pre-treinadas no ImageNet)
       |
       v
  Features (512 valores)
       |
       v
  Linear(512 -> 4)  -- "Dressing Classico"
       |
       v
[CIRCUITO QUANTICO - 4 Qubits]
  |
  |-- AngleEmbedding: converte os 4 valores em angulos de rotacao dos qubits
  |-- StronglyEntanglingLayers (x q_depth): cria emaranhamento entre os qubits
  |-- Medicao PauliZ em cada qubit: retorna 4 valores reais
  |
       |
       v
  Medicoes quanticas (4 valores reais)
       |
       v
  Linear(4 -> 2)
       |
       v
  Logits [Benigno, Maligno]
```

**AngleEmbedding:** Codifica dados classicos em qubits usando rotacoes. Um valor de 0.5 vira uma rotacao de 0.5 radianos no qubit correspondente. E a forma mais natural de inserir informacao classica em um circuito quantico.

**StronglyEntanglingLayers:** E o "coracao" do VQC. Cada camada aplica rotacoes parametrizadas (os pesos treinaveis quanticos) em cada qubit e depois cria CNOT gates entre eles para gerar emaranhamento. O emaranhamento e a propriedade quantica que permite ao circuito criar correlacoes nao-classicas entre os qubits.

**Medicao PauliZ:** Ao medir o valor esperado do operador Pauli-Z em cada qubit, obtemos um numero real entre -1 e +1. Esses 4 numeros representam o estado final do circuito apos o processamento quantico.

### Diferenciacao Automatica Quantica (Adjoint Method)
Para treinar pesos classicos e quanticos simultaneamente (end-to-end), precisamos calcular gradientes em relacao a todos os parametros.

- **Para os pesos classicos:** O PyTorch usa Backpropagation padrao (Regra da Cadeia).
- **Para os pesos quanticos:** O PennyLane usa o metodo **Adjoint Differentiation**. Ele explora o fato de que operacoes quanticas sao matrizes unitarias (e portanto invessiveis/adjuntas) para calcular os gradientes de todos os parametros quanticos em uma unica passagem, sem precisar rodar o circuito multiplas vezes como o metodo Parameter-Shift Rule (PSA) exige.

O Adjoint e especialmente eficiente em **simuladores classicos** (como o `default.qubit` do PennyLane), onde temos acesso direto ao estado interno do circuito.

---

## Gerenciamento e Persistencia de Modelos

### Por que Joblib e nao apenas PyTorch?
O PyTorch salva modelos em arquivos `.pth` (state_dict). Isso e simples para modelos puramente classicos. Porem, o modelo hibrido contem camadas do PennyLane (TorchLayer) que nao sao diretamente serializaveis pelo sistema de pickle padrao do Python.

A solucao adotada salva em `joblib` um dicionario contendo:
- `model_state_dict`: Os pesos do modelo extraidos manualmente via `state_dict()`.
- `architecture`: Os metadados necessarios para reconstruir a arquitetura (n_qubits, q_depth).
- `hyperparams`: Os hiperparametros usados no treinamento.
- `timestamp`: Data e hora do salvamento.

Para carregar o modelo novamente, basta recriar a arquitetura com os metadados salvos e carregar o state_dict nos pesos.

---

## Estrutura de Arquivos dos Resultados

```
data/group_works/group_01/
|
|-- best_params_classical.json       (melhores hiperparametros do classico)
|-- best_params_hybrid.json          (melhores hiperparametros do hibrido)
|-- optimization_time.txt            (tempo da fase de otimizacao)
|
|-- optuna/
|   |-- best_classical_checkpoint.pth   (melhor modelo classico do Optuna)
|   |-- best_hybrid_checkpoint.pth      (melhor modelo hibrido do Optuna)
|
|-- kfold/
|   |-- kfold_report.txt             (relatorio consolidado de todos os folds)
|
|-- pipeline_results/
|   |-- comparacao_modelos.png       (curvas de loss e acuracia)
|   |-- confusion_matrix_classical.png
|   |-- confusion_matrix_hybrid.png
|   |-- Classical_Final_metrics.txt  (metricas do treinamento final classico)
|   |-- Hybrid_Final_metrics.txt     (metricas do treinamento final hibrido)
|   |-- pipeline_summary.txt         (resultado do teste cego + tempo total)
|
|-- trained_models/
|   |-- classical_model.joblib       (modelo classico final pronto para inferencia)
|   |-- hybrid_model.joblib          (modelo hibrido final pronto para inferencia)
|
|-- notebook/
|   |-- comparacao_modelos.png       (graficos gerados pelo notebook)
|   |-- confusion_matrix_*.png
|   |-- notebook_summary.txt
|
|-- mri_dataset/                     (imagens originais - nao remover)
```

---

## Decisoes de Design e Justificativas

| Decisao | Alternativa Considerada | Por que esta escolha? |
|---|---|---|
| ResNet-18 como backbone | Treinar CNN do zero | Dados insuficientes para treinar do zero sem overfitting severo |
| Backbone congelado | Fine-tuning completo | Evita overfitting; acelera treinamento |
| 4 Qubits | Mais qubits | Custo computacional cresce exponencialmente com qubits |
| Adjoint Differentiation | Parameter-Shift Rule | 10-20x mais rapido em simuladores |
| Optuna com 10 trials | Grid Search | Mais eficiente; busca dirigida pelos resultados anteriores |
| 5 folds no K-Fold | 10 folds | Bom equilibrio entre custo e confianca estatistica |
| 20 epocas no treino final | Mais epocas | Checkpointing garante que o modelo nao piore; mais epocas nao ajudam |
| Divisao 70/15/15 | 80/20 sem teste cego | Conjunto de teste cego garante avaliacao imparcial |
| Batch size 16 no treino final | Batch maior | Uso de memoria moderado; generalizacao ligeiramente melhor |
