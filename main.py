import pandas as pd
import datetime
from pycaret.classification import setup, create_model, pull, models
from codecarbon import EmissionsTracker
from sklearn.datasets import make_classification
import time

# --- 0. CONFIGURAÇÃO DO EXPERIMENTO ---
EXECUTION_MODE = 'GPU_PER_MODEL_CC_MODE_OFF'

# --- 1. GERAR E PREPARAR OS DADOS ---
print("Gerando o conjunto de dados...")
X, y = make_classification(
    n_samples=200000,
    n_features=50,
    n_informative=25,
    n_redundant=5,
    random_state=42
)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y_s = pd.Series(y, name='target')
df = pd.concat([X_df, y_s], axis=1)
print("Dados prontos.")

# --- 2. CONFIGURAR O AMBIENTE PYCARET (FEITO APENAS UMA VEZ) ---
print("Configurando o ambiente PyCaret com use_gpu=True...")
clf_setup = setup(data=df, target='target', session_id=123, use_gpu=True, verbose=False)

# --- 3. EXECUTAR O BENCHMARK POR MODELO ---
models_to_test = ['lr', 'knn', 'nb', 'dt', 'rf', 'xgboost', 'lightgbm'] 
all_results = []

print(f"Iniciando benchmark para {len(models_to_test)} modelos...")

for model_id in models_to_test:
    print(f"--- Processando modelo: {model_id} ---")
    tracker = EmissionsTracker(project_name=f"{EXECUTION_MODE}_{model_id}")
    
    try:
        tracker.start()
        create_model(model_id, verbose=False)
        time.sleep(1) 
    except Exception as e:
        print(f"Falha ao treinar o modelo {model_id}. Erro: {e}")
    finally:
        # Apenas para o tracker para obter a medição
        tracker.stop()

    # CORRIGIDO: Acessa os dados a partir do atributo '.final_emissions_data'
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
        all_results.append(combined_results)

print("\nBenchmark individual concluído.")

# --- 4. SALVAR OS RESULTADOS FINAIS ---
if all_results:
    final_df = pd.DataFrame(all_results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"final_results_per_model_{EXECUTION_MODE}_{timestamp}.csv"
    
    first_cols = ['model_id', 'duration_seconds', 'energy_consumed_kWh', 'gpu_energy_kWh', 'Accuracy', 'AUC', 'F1']
    other_cols = [col for col in final_df.columns if col not in first_cols]
    final_df = final_df[first_cols + other_cols]
    
    final_df.to_csv(final_filename, index=False)
    
    print(f"\nResultados combinados salvos em: {final_filename}")
    print("\n--- Amostra do Relatório Final ---")
    print(final_df[['model_id', 'duration_seconds', 'gpu_energy_kWh', 'Accuracy', 'F1']].head())
else:
    print("Nenhum resultado foi coletado.")
