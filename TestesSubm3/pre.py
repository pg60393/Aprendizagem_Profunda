# pre_processamento.py

import pandas as pd

def criar_super_dataset():
    print("A carregar os datasets...")
    # 1. Carregar os datasets (Ajusta o nome para o CSV que descarregaste)
    df_original = pd.read_csv('dataset_final.csv', sep=';')
    df_kaggle = pd.read_csv('data.csv') 

    # 2. O mapeamento mágico (Ajustar as chaves da esquerda conforme o dataset)
    mapeamento = {
        'Human': 'Human',
        'GPT-4': 'OpenAI',
        'GPT-3.5': 'OpenAI',
        'Claude-v1': 'Anthropic',
        'LLaMA-70B': 'Meta',
        'PaLM': 'Google',
        'Gemini': 'Google'
    }

    # 3. Limpeza e Formatação
    print("A mapear e limpar os dados...")
    df_kaggle['Label'] = df_kaggle['source'].map(mapeamento)
    df_kaggle['Text'] = df_kaggle['text']
    df_kaggle = df_kaggle.dropna(subset=['Label'])
    df_kaggle = df_kaggle[['Text', 'Label']]

    # 4. Equilibrar e Juntar
    print("A equilibrar e fundir com o dataset original...")
    df_equilibrado = df_kaggle.groupby('Label').sample(n=800, random_state=42)
    df_super = pd.concat([df_original, df_equilibrado], ignore_index=True)
    
    # 5. Baralhar e Guardar
    df_super = df_super.sample(frac=1, random_state=42).reset_index(drop=True)
    df_super.to_csv('super_dataset_final.csv', index=False, sep=';')
    
    print("✅ 'super_dataset_final.csv' gerado com sucesso e pronto a usar no moda.ipynb!")

if __name__ == "__main__":
    criar_super_dataset()