# Aprendizagem Profunda (Trabalho Prático)

**Universidade do Minho 2026**

## Constituição do Grupo (Grupo 6)

| Número | Nome |
|----------|-----------|
| A100599 | Gonçalo Antunes Corais |
| A98639 | Bruno Gonçalo Costa Campos |
| PG60393 | Simão Novais Vieira da Silva |

## Organização do Repositório (Resumo de Navegação)

### Estrutura principal
- `Aprendizagem_Profunda/`
  - `tarefa3_pytorch.py` : Ficheiro de implementação em PyTorch relativo à tarefa 3.
  - `pre_processamento.ipynb` : Notebook com análise e pré-processamento dos dados.
  - `datasets/` : Ficheiros CSV de dados usados para as tarefas.
  - `numpy/` : Implementações de rede neural feitas a partir de zero com NumPy.
    - `neuralnet.py`, `layers.py`, `activation.py`, `losses.py`, `optimizer.py`, `metrics.py`.

- `Subm1/`
  - Pasta com os ficheiros necessários para a entrega do trabalho correspondentes à 1ª submissão
  - Inclui os ficheiros `subm1-g6-MIA-A.csv`, `subm1-g6-MIA-B.csv` e notebooks com os 2 melhores modelos treinados pelo grupo.

### Como correr
1. Executar [pre_processamento.ipynb](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/pre_processamento.ipynb) para a criação dos ficheiros .npy de treino, validação e teste.
2. Entrar na pasta `numpy/`. `python cd numpy`
3. Executar [tarefa2_main.py](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/numpy/tarefa2_main.py) para treinar e avaliar o modelo em NumPy. `python tarefa2_main.py`
4. Sair da pasta `numpy/`. `cd ..`
5. Executar [tarefa3_pytorch.py](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/tarefa3_pytorch.py) para treinar e avaliar o modelo em PyTorch. `python tarefa3_pytorch.py`
6. Entrar na pasta `Subm1/`. `cd Subm1`
7. Executar [subm1-g6-MIA-A.ipynb](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/Subm1/subm1-g6-MIA-A.ipynb).
8. Executar [subm1-g6-MIA-B.ipynb](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/Subm1/subm1-g6-MIA-B.ipynb).
