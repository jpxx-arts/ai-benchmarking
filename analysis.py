import pandas as pd
from pathlib import Path


def analyze_results(results_path, mode):
    all_files = list(results_path.glob('*.csv'))
    if not all_files:
        return None

    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    # Agrupa modelo e calcula a média e desvio padrão para colunas numéricas
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols.remove('run')  # Não agrega a coluna 'run'

    grouped = df.groupby('model_id')[numeric_cols]

    stats_df = grouped.agg(['mean', 'std'])

    # Exemplo: ('duration_seconds', 'mean') para 'duration_seconds_mean')
    stats_df.columns = ['_'.join(col).strip()
                        for col in stats_df.columns.values]
    # Adiciona sufixo do modo (ex: _CC_MODE_OFF)
    stats_df = stats_df.add_suffix(f'_{mode}')

    return stats_df


def main():
    base_path = Path('results')
    off_path = base_path / 'CC_MODE_OFF'
    on_path = base_path / 'CC_MODE_ON'

    print("Analisando resultados para o modo CC OFF...")
    stats_off = analyze_results(off_path, 'OFF')

    print("Analisando resultados para o modo CC ON...")
    stats_on = analyze_results(on_path, 'ON')

    if stats_off is None or stats_on is None:
        print("Erro: Pastas de resultados não encontradas ou vazias. \
              Execute o main.py primeiro para ambos os modos.")
        return

    # Junta os resultados dos dois modos em uma única tabela
    final_report = pd.merge(stats_off, stats_on, on='model_id', how='outer')

    # --- Cálculo do Overhead ---
    metrics_for_overhead = [
        'duration_seconds',
        'energy_consumed_kWh',
        'gpu_energy_kWh',
        'cpu_energy_kWh'
    ]

    for metric in metrics_for_overhead:
        mean_off = f'{metric}_mean_OFF'
        mean_on = f'{metric}_mean_ON'
        overhead_col = f'{metric}_overhead_%'

        if mean_off in final_report.columns and \
           mean_on in final_report.columns:
            # Fórmula: ((ON - OFF) / OFF) * 100
            final_report[overhead_col] = (
                (final_report[mean_on] - final_report[mean_off]) /
                final_report[mean_off]) * 100

    # --- Organização e Salvamento ---
    # Reorganiza as colunas para melhor visualização
    cols_order = [
        'duration_seconds_mean_OFF', 'duration_seconds_std_OFF',
        'duration_seconds_mean_ON', 'duration_seconds_std_ON',
        'duration_seconds_overhead_%',
        'gpu_energy_kWh_mean_OFF', 'gpu_energy_kWh_std_OFF',
        'gpu_energy_kWh_mean_ON', 'gpu_energy_kWh_std_ON',
        'gpu_energy_kWh_overhead_%',
        'Accuracy_mean_OFF', 'Accuracy_mean_ON'
    ]
    # Adiciona colunas restantes que não estão na lista de ordem
    other_cols = [col for col in final_report.columns if col not in cols_order]
    final_report = final_report[cols_order + other_cols]

    output_filename = 'final_summary_report.csv'
    final_report.to_csv(output_filename)

    print(f"\nAnálise concluída! Relatório final salvo em: {output_filename}")
    print("\n--- Amostra do Relatório Final ---")
    print(final_report[cols_order].head())


if __name__ == '__main__':
    main()
