import pandas as pd
from pathlib import Path

# Função de Sliding Window
def sliding_window(df, window_size):
    for i in range(len(df) - window_size + 1):
        yield df.iloc[i:i + window_size]

# Importação da Série Temporal Estática
parent_path = Path(__file__).parent
data_path = (parent_path / "../Data/VilaSol2024.csv").resolve()
print(data_path)

df = pd.read_csv(data_path)
df['Data e Hora'] = pd.to_datetime(df['Data e Hora'])

# Lógica para Simulação de Streaming 
tamanho_janela = 2016 # Tamanho da Janela Considerada (e.g., 7 dias de dados com frequência de aquisição de 5-5 min (12 registos hora * 24 horas * 7 dias) = 2016 Pontos)

for i, janela in enumerate(sliding_window(df, tamanho_janela)):
    print(f"Janela {i+1}: {janela} registos")

    # Os Cálculos São Realizados Aqui...