import pandas as pd
import datetime
import os
import time
import sys
from pycaret.classification import setup, create_model, pull
from codecarbon import EmissionsTracker
from sklearn.datasets import make_classification

# --- 0. CONFIGURAÇÕES GERAIS ---
# O modo agora é definido pelo script 'run_experiment.sh'
EXECUTION_MODE = 'CC_MODE_ON'
N_RUNS = 1

# --- 1. LEITURA DO CENÁRIO E GERAÇÃO DE DADOS ---
if len(sys.argv) != 2:
    print("Erro: Forneça o cenário de custo ('low_cost' ou 'high_cost') como argumento.")
    sys.exit(1)

cost_scenario = sys.argv[1]

print(f"Gerando o conjunto de dados para o cenário: {cost_scenario}")

if cost_scenario == "low_cost":
    X, y = make_classification(
        n_samples=20000, n_features=50,
        n_informative=25, n_redundant=5, random_state=42
    )
elif cost_scenario == "high_cost":
    X, y = make_classification(
        n_samples=2000000, n_features=100,
        n_informative=50, n_redundant=10, random_state=42
    )
else:
    print(f"Erro: Cenário '{cost_scenario}' desconhecido.")
    sys.exit(1)

X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y_s = pd.Series(y, name='target')
df = pd.concat([X_df, y_s], axis=1)
print("Dados prontos.")

# --- 2. SETUP DO PYCARET ---
clf_setup = setup(data=df, target='target', session_id=123,
                  use_gpu=True, verbose=False)

# --- 3. EXECUÇÃO DO BENCHMARK ---
output_dir = os.path.join('results', EXECUTION_MODE)
os.makedirs(output_dir, exist_ok=True)
models_to_test = ['lr', 'knn', 'nb', 'dt', 'rf', 'xgboost']

all_results_for_this_run = []

for model_id in models_to_test:
    print(f"--- Processando modelo: {model_id} (Cenário: {cost_scenario}) ---")

    tracker = EmissionsTracker(
        project_name=f"{EXECUTION_MODE}_{cost_scenario}_{model_id}"
    )

    try:
        tracker.start()
        create_model(model_id, verbose=False)
        time.sleep(1)
    except Exception as e:
        print(f"Falha ao treinar o modelo {model_id}. Erro: {e}")
    finally:
        tracker.stop()

    emissions_data = tracker.final_emissions_data
    performance_df = pull()
    mean_performance = performance_df.loc['Mean'].to_dict()

    if emissions_data:
        sustainability_metrics = {
            'model_id': model_id,
            'duration_seconds': emissions_data.duration,
            'energy_consumed_kWh': emissions_data.energy_consumed,
            'emissions_kg_CO2eq': emissions_data.emissions,
            'cpu_energy_kWh': emissions_data.cpu_energy,
            'gpu_energy_kWh': emissions_data.gpu_energy,
            'ram_energy_kWh': emissions_data.ram_energy
        }
        combined_results = {**sustainability_metrics, **mean_performance}
        all_results_for_this_run.append(combined_results)

if all_results_for_this_run:
    run_df = pd.DataFrame(all_results_for_this_run)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_filename = os.path.join(output_dir, f"{EXECUTION_MODE}_{cost_scenario}_{timestamp}.csv")
    run_df.to_csv(run_filename, index=False)
    print(f"Resultados salvos em: {run_filename}")
