import pandas as pd
import datetime
import os
import time
import sys
from pycaret.classification import setup, create_model, pull
from codecarbon import EmissionsTracker
from sklearn.datasets import make_classification

EXECUTION_MODE = "PLACEHOLDER"

# --- 1. LEITURA DOS ARGUMENTOS E GERAÇÃO DE DADOS ---
if len(sys.argv) < 3:
    print("Erro: Forneça o cenário ('comm_bound' ou 'compute_bound') e o run_id como argumentos.")
    print("Exemplo: python main.py comm_bound 1")
    sys.exit(1)

cost_scenario = sys.argv[1]
run_id = int(sys.argv[2])
print(f"Execução ID: {run_id}")
print(f"Gerando o conjunto de dados para o cenário: {cost_scenario}")

if cost_scenario == "comm_bound":
    # Cenário de comunicação: MUITAS LINHAS E MUITAS COLUNAS.
    # O volume total de dados é o gargalo.
    print("Gerando dataset LARGO e LONGO para gargalo de comunicação...")
    X, y = make_classification(
        n_samples=200000,       # Mais linhas
        n_features=100,         # Mais colunas
        n_informative=40,       # Sinal de base
        n_redundant=20,         # Colunas "inúteis" para inflar o volume
        random_state=42
    )
elif cost_scenario == "compute_bound":
    # Cenário de computação: MENOS LINHAS E MENOS COLUNAS.
    # A complexidade do modelo sobre um dataset "limpo" é o gargalo.
    print("Gerando dataset MENOR e mais DENSO para gargalo computacional...")
    X, y = make_classification(
        n_samples=50000,        # Menos linhas
        n_features=40,          # Menos colunas
        n_informative=30,       # Alta proporção de sinal útil
        n_redundant=5,          # Pouco "ruído"
        random_state=42
    )
else:
    print(f"Erro: Cenário '{cost_scenario}' desconhecido.")
    sys.exit(1)

X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y_s = pd.Series(y, name='target')
df = pd.concat([X_df, y_s], axis=1)
print("Dados prontos.")

# --- 2. SETUP DO PYCARET ---
# Esta etapa já consome recursos, especialmente com dados grandes
print("Configurando o ambiente PyCaret com use_gpu=True...")
clf_setup = setup(data=df, target='target', session_id=123, use_gpu=True, verbose=False)

# Estes parâmetros forçarão uma carga de trabalho computacional massiva
compute_bound_params = {
    'lr': {'max_iter': 2000},
    'knn': {'n_neighbors': 100},
    'dt': {'max_depth': 30},
    'rf': {'n_estimators': 1500, 'max_depth': 20, 'n_jobs': -1},
    'xgboost': {'n_estimators': 1500, 'max_depth': 12, 'tree_method': 'gpu_hist'}
}

# --- 4. EXECUÇÃO DO BENCHMARK ---
output_dir = os.path.join('results', EXECUTION_MODE)
os.makedirs(output_dir, exist_ok=True)
models_to_test = ['lr', 'knn', 'nb', 'dt', 'rf', 'xgboost']

all_results_for_this_run = []

for model_id in models_to_test:
    print(f"--- Processando modelo: {model_id} (Cenário: {cost_scenario}) ---")
    
    tracker = EmissionsTracker(
        project_name=f"{EXECUTION_MODE}_{cost_scenario}_{model_id}",
        # Se você tiver problemas, pode remover o output_file e output_dir
        # output_file=f"emissions_{model_id}.csv",
        # output_dir=output_dir
    )
    
    try:
        tracker.start()
        
        # Lógica para aplicar os parâmetros de alto custo
        if cost_scenario == 'compute_bound' and model_id in compute_bound_params:
            print(f"    -> Usando hiperparâmetros de alto custo para {model_id}")
            params = compute_bound_params[model_id]
            create_model(model_id, verbose=False, **params)
        else:
            # Para o cenário 'comm_bound' (dados grandes, modelo simples)
            # ou modelos sem params customizados (como 'nb')
            print(f"    -> Usando parâmetros padrão para {model_id}")
            create_model(model_id, verbose=False)

        time.sleep(1)
    except Exception as e:
        print(f"Falha ao treinar o modelo {model_id}. Erro: {e}")
    finally:
        emissions_data = tracker.stop()

    # Coleta de resultados
    performance_df = pull()
    if not performance_df.empty:
        mean_performance = performance_df.loc['Mean'].to_dict()
        
        if emissions_data:
            sustainability_metrics = {
                'model_id': model_id,
                'run_id': run_id,
                'duration_seconds': emissions_data,
                'energy_consumed_kWh': tracker.final_emissions_data.energy_consumed,
                'emissions_kg_CO2eq': tracker.final_emissions_data.emissions,
                'cpu_energy_kWh': tracker.final_emissions_data.cpu_energy,
                'gpu_energy_kWh': tracker.final_emissions_data.gpu_energy,
                'ram_energy_kWh': tracker.final_emissions_data.ram_energy
            }
            combined_results = {**sustainability_metrics, **mean_performance}
            all_results_for_this_run.append(combined_results)

if all_results_for_this_run:
    run_df = pd.DataFrame(all_results_for_this_run)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_filename = os.path.join(output_dir, f"{EXECUTION_MODE}_{cost_scenario}_{timestamp}.csv")
    run_df.to_csv(run_filename, index=False)
    print(f"\nResultados salvos em: {run_filename}")
