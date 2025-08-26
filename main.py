import pandas as pd
import datetime
from pycaret.classification import setup, compare_models, pull
from codecarbon import EmissionsTracker
from sklearn.datasets import make_classification

# --- 0. CONFIGURAÇÃO DO EXPERIMENTO ---
# Altere para 'CC_MODE_ON' no ambiente confidencial
EXECUTION_MODE = 'GPU_BENCHMARK_CC_MODE_OFF' 

# --- 1. GERAR E PREPARAR OS DADOS ---
print("Gerando o conjunto de dados...")
X, y = make_classification(
    n_samples=500, # Manter um dataset grande para a carga de trabalho ser relevante
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)
# PyCaret funciona melhor com DataFrames do Pandas
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y_s = pd.Series(y, name='target')
df = pd.concat([X_df, y_s], axis=1)
print("Dados prontos.")

# --- 2. CONFIGURAR O TRACKER DE EMISSÕES ---
tracker = EmissionsTracker(project_name=f"gpu_benchmark_{EXECUTION_MODE}")

# --- 3. EXECUTAR O BENCHMARK ---
try:
    print(f"Iniciando o benchmark no modo: {EXECUTION_MODE}...")
    tracker.start()
    
    # --- A MÁGICA DO PYCARET ---
    # 1. Configurar o ambiente. É aqui que ativamos a GPU.
    print("Configurando o ambiente PyCaret com use_gpu=True...")
    clf_setup = setup(data=df, target='target', session_id=123, use_gpu=True, verbose=False)
    
    # 2. Comparar os modelos. Esta função treina e avalia uma lista de modelos.
    print("Iniciando a comparação de modelos na GPU...")
    best_model = compare_models()
    
    # 3. Puxar a tabela de resultados
    results_df = pull()
    
    print("Benchmark do PyCaret concluído.")

finally:
    tracker.stop()
    print("Tracker do CodeCarbon finalizado.")

# --- 4. SALVAR OS RESULTADOS AUTOMATICAMENTE ---
if tracker.final_emissions_data:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar resultados do PyCaret
    pycaret_filename = f"pycaret_results_{EXECUTION_MODE}_{timestamp}.csv"
    results_df.to_csv(pycaret_filename)
    print(f"\nResultados de desempenho salvos em: {pycaret_filename}")
    
    # Extrai o objeto com os dados detalhados
    emissions_data = tracker.final_emissions_data
    
    emissions_df = pd.DataFrame({
        'timestamp': [emissions_data.timestamp],
        'project_name': [emissions_data.project_name],
        'duration_seconds': [emissions_data.duration],
        'energy_consumed_kWh': [emissions_data.energy_consumed],
        'emissions_kg_CO2eq': [emissions_data.emissions],
        'cpu_energy_kWh': [emissions_data.cpu_energy],
        'gpu_energy_kWh': [emissions_data.gpu_energy],
        'ram_energy_kWh': [emissions_data.ram_energy]
    })
    
    codecarbon_filename = f"codecarbon_results_{EXECUTION_MODE}_{timestamp}.csv"
    emissions_df.to_csv(codecarbon_filename, index=False)
    print(f"Resultados de emissões salvos em: {codecarbon_filename}")

    print("\n--- Resumo Final ---")
    print(f"Consumo total de energia: {emissions_data.energy_consumed:.6f} kWh")
    print("Tabela de Desempenho (PyCaret):")
    print(results_df.head())
else:
    print("Não foi possível coletar os dados de emissões.")
