import pandas as pd
from pathlib import Path
import re


def analyze_results(results_path, mode, scenario):
    """Lê os CSVs de um cenário específico, calcula média e desvio padrão."""
    all_files = list(results_path.glob(f'*{scenario}*.csv'))

    if not all_files:
        print(
            f"    - Nenhum arquivo encontrado para o modo '{mode}' e cenário '{scenario}'.")
        return None

    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if 'run' in numeric_cols:
        numeric_cols.remove('run')

    grouped = df.groupby('model_id')[numeric_cols]
    stats_df = grouped.agg(['mean', 'std'])

    stats_df.columns = ['_'.join(col).strip()
                        for col in stats_df.columns.values]
    stats_df = stats_df.add_suffix(f'_{mode}')

    return stats_df


def main():
    """Função principal para rodar a análise completa para todos os cenários."""
    base_path = Path('results')
    off_path = base_path / 'CC_MODE_OFF'
    on_path = base_path / 'CC_MODE_ON'

    if not off_path.is_dir() or not on_path.is_dir():
        print("Erro: As pastas 'results/CC_MODE_OFF' e/ou 'results/CC_MODE_ON' não foram encontradas.")
        return

    all_filenames = [f.name for f in off_path.glob('*.csv')]
    scenarios = sorted(list(
        set(re.findall(r'_(low_cost|medium_cost|high_cost)_', ' '.join(all_filenames)))))

    if not scenarios:
        print("Erro: Nenhum cenário (low_cost, high_cost, etc.) encontrado nos nomes dos arquivos.")
        return

    print(f"Cenários detectados: {scenarios}")

    all_scenario_reports = []

    for scenario in scenarios:
        print(f"\n--- Analisando Cenário: {scenario} ---")

        stats_off = analyze_results(off_path, 'OFF', scenario)
        stats_on = analyze_results(on_path, 'ON', scenario)

        if stats_off is None or stats_on is None:
            print(
                f"    - Pulando o cenário '{scenario}' por falta de dados em um dos modos.")
            continue

        final_report = pd.merge(stats_off, stats_on,
                                on='model_id', how='outer')

        metrics_for_overhead = [
            'duration_seconds', 'energy_consumed_kWh', 'gpu_energy_kWh', 'cpu_energy_kWh']
        for metric in metrics_for_overhead:
            mean_off = f'{metric}_mean_OFF'
            mean_on = f'{metric}_mean_ON'
            overhead_col = f'{metric}_overhead_%'

            if mean_off in final_report.columns and mean_on in final_report.columns:
                final_report[overhead_col] = (
                    (final_report[mean_on] - final_report[mean_off]) / final_report[mean_off]) * 100

        final_report['scenario'] = scenario
        all_scenario_reports.append(final_report)

    if all_scenario_reports:
        full_final_report = pd.concat(all_scenario_reports)

        # <<< LINHA DA CORREÇÃO AQUI >>>
        # Converte o índice 'model_id' de volta para uma coluna.
        full_final_report.reset_index(inplace=True)

        # Agora a reorganização de colunas e o print funcionarão corretamente
        cols = full_final_report.columns.tolist()
        cols.insert(0, cols.pop(cols.index('scenario')))
        # A linha abaixo foi corrigida para usar a nova coluna 'model_id'
        cols.insert(1, cols.pop(cols.index('model_id')))
        full_final_report = full_final_report.loc[:, cols]

        output_filename = 'final_summary_report_with_scenarios.csv'
        # 'index=False' para não salvar o índice numérico
        full_final_report.to_csv(output_filename, index=False)

        print(f"\nAnálise completa concluída! Relatório final salvo em: {output_filename}")
        print("\n--- Amostra do Relatório Final ---")
        # Esta linha agora funciona, pois 'model_id' é uma coluna
        print(full_final_report[['scenario', 'model_id', 'duration_seconds_mean_OFF', 'duration_seconds_mean_ON', 'duration_seconds_overhead_%']].head())
    else:
        print("\nNenhuma análise pôde ser concluída.")


if __name__ == '__main__':
    main()
