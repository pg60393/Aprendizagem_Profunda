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
  - `dataset_final.csv` : Ficheiro CSV de dados usados para as tarefas.
  - `SVC.py.` : Ficheiro de implementação do modelo SVC.

- `Subm1/`
  - Pasta com os ficheiros necessários para a entrega do trabalho correspondentes à 1ª submissão
  - Inclui os ficheiros `subm1-g6-MIA-A.csv`, `subm1-g6-MIA-B.csv` e notebooks com os 2 melhores modelos treinados pelo grupo.
  
- `Subm2/`
  - Pasta com os ficheiros necessários para a entrega do trabalho correspondentes à 2ª submissão
  - Inclui os ficheiros `subm2-g6-MIA-A.csv`, `subm2-g6-MIA-B.csv` e notebooks com os 2 melhores modelos treinados pelo grupo.

### Como correr
1. Executar [pre_processamento.ipynb](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/pre_processamento.ipynb) para a criação dos ficheiros .npy de treino, validação e teste.
2. Executar [SVC.py](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/SVC.py) para treinar e avaliar o modelo SVC. `python SVC.py`
3. Executar [tarefa3_pytorch.py](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/tarefa3_pytorch.py) para treinar e avaliar o modelo em PyTorch. `python tarefa3_pytorch.py`
4. Entrar na pasta `Subm2/`. `cd Subm2`
5. Executar [subm2-g6-MIA-A.ipynb](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/Subm2/subm2-g6-MIA-A.ipynb).
6. Executar [subm2-g6-MIA-B.ipynb](https://github.com/pg60393/Aprendizagem_Profunda/blob/main/Subm2/subm2-g6-MIA-B.ipynb).
