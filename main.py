import pandas as pd
import datetime
import os
import time
from pycaret.classification import setup, create_model, pull
from codecarbon import EmissionsTracker
from sklearn.datasets import make_classification

# --- 0. CONFIGURAÇÕES PRINCIPAIS DO EXPERIMENTO ---

EXECUTION_MODE = 'CC_MODE_OFF'
N_RUNS = 30

# --- 1. GERAÇÃO DO CONJUNTO DE DADOS ---
print("Gerando o conjunto de dados...")
# y é o vetor de rótulos
X, y = make_classification(
    n_samples=200000,  # Número de amostras (dados)
    n_features=50,  # Número de características para classificação
    n_informative=25,  # Apenas 25 características são relevantes
    n_redudant=5,  # Características linearmente dependentes
    random_state=42  # "Seed" para reprodutibilidade em experimentos
)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y_s = pd.Series(y, name='target')
df = pd.concat([X_df, y_s], axis=1)
print("Dados prontos.")

# --- 2. SETUP DO PYCARET ---
print("Configurando o ambiente PyCaret")
clf_setup = setup(data=df, target='target', session_id=123,
                  use_gpu=True, verbose=False)

# --- 3. EXECUÇÃO DO BENCHMARK ---
# Cria as pastas de resultado final e de logs brutos
final_results_dir = os.path.join('results', EXECUTION_MODE)
codecarbon_log_dir = os.path.join('codecarbon_logs', EXECUTION_MODE)
os.makedirs(final_results_dir, exist_ok=True)
os.makedirs(codecarbon_log_dir, exist_ok=True)

models_to_test = [
    'lr',      # Logistic Regression
    'knn',     # K-Nearest Neighbors
    'nb',      # Naive Bayes
    'dt',      # Decision Tree
    'rf',      # Random Forest
    'xgboost'  # Extreme Gradient Boosting
]

for run in range(1, N_RUNS + 1):
    print(f"\n======= INICIANDO EXECUÇÃO {
          run}/{N_RUNS} PARA O MODO {EXECUTION_MODE} =======")

    all_results_for_this_run = []

    for model_id in models_to_test:
        print(f"--- Processando modelo: {model_id} (Execução {run}) ---")

        tracker = EmissionsTracker(
            project_name=f"{EXECUTION_MODE}_{model_id}_run_{run}",
            output_dir=codecarbon_log_dir
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
                'run': run,
                'duration_seconds': emissions_data.duration,
                'energy_consumed_kWh': emissions_data.energy_consumed,
                'emissions_kg_CO2eq': emissions_data.emissions,
                'cpu_energy_kWh': emissions_data.cpu_energy,
                'gpu_energy_kWh': emissions_data.gpu_energy,
                'ram_energy_kWh': emissions_data.ram_energy
            }
            combined_results = {**sustainability_metrics, **mean_performance}
            all_results_for_this_run.append(combined_results)

    # Salva o resultado processado desta execução em um arquivo CSV único
    if all_results_for_this_run:
        run_df = pd.DataFrame(all_results_for_this_run)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_filename = os.path.join(final_results_dir, f"run_{run}_{
                                    EXECUTION_MODE}_{timestamp}.csv")
        run_df.to_csv(run_filename, index=False)
        print(f"Resultados da execução {run} salvos em: {run_filename}")

print(f"\n======= FIM DO BENCHMARK PARA O MODO {EXECUTION_MODE} =======")
