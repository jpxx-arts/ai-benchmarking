import pandas as pd
import os
from pathlib import Path
import re

def detect_outliers_iqr(group, column='duration_seconds', threshold=1.5):
    """
    Identifica outliers em um grupo de dados usando o método IQR.
    Retorna o grupo original com uma nova coluna booleana 'is_outlier'.
    """
    Q1 = group[column].quantile(0.25)
    Q3 = group[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (group[column] < lower_bound) | (group[column] > upper_bound)
    group['is_outlier'] = outliers
    return group

def analyze_results(full_df, mode, scenario):
    """Filtra por modo/cenário e calcula média/desvio padrão."""
    df = full_df[(full_df['scenario'] == scenario) & (full_df['mode'] == mode)].copy()
    
    if df.empty:
        print(f"    - Nenhum dado encontrado para o modo '{mode}' e cenário '{scenario}'.")
        return None
        
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    # <<< MUDANÇA 1: Verificação de segurança >>>
    # Só tenta remover 'run' se a coluna existir nos dados
    if 'run' in numeric_cols:
        numeric_cols.remove('run')

    if 'is_outlier' in df.columns:
        df_filtered = df[df['is_outlier'] == False]
        if len(df) > len(df_filtered):
             print(f"    - {len(df) - len(df_filtered)} outlier(s) removido(s) dos cálculos para o modo {mode}.")
    else:
        df_filtered = df

    grouped = df_filtered.groupby('model_id')[numeric_cols]
    stats_df = grouped.agg(['mean', 'std'])
    
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    stats_df = stats_df.add_suffix(f'_{mode}')
    
    return stats_df

def main():
    """Função principal para rodar a análise completa."""
    base_path = Path('results')
    off_path = base_path / 'CC_MODE_OFF'
    on_path = base_path / 'CC_MODE_ON'
    
    if not off_path.is_dir() or not on_path.is_dir():
        print("Erro: As pastas 'results/CC_MODE_OFF' e/ou 'results/CC_MODE_ON' não foram encontradas.")
        return

    all_runs = []
    for mode_path, mode_name in [(off_path, 'OFF'), (on_path, 'ON')]:
        for csv_file in mode_path.glob('*.csv'):
            df = pd.read_csv(csv_file)
            df['mode'] = mode_name
            scenario_match = re.search(r'_(low_cost|medium_cost|high_cost)_', csv_file.name)
            if scenario_match:
                df['scenario'] = scenario_match.group(1)
            all_runs.append(df)

    if not all_runs:
        print("Nenhum arquivo de resultado encontrado.")
        return
        
    full_df = pd.concat(all_runs, ignore_index=True)

    print("\n--- Verificando Outliers (baseado em duration_seconds) ---")
    
    full_df['is_outlier'] = full_df.groupby(['scenario', 'mode', 'model_id'])['duration_seconds'].transform(lambda s: (s < s.quantile(0.25) - 1.5 * (s.quantile(0.75) - s.quantile(0.25))) | (s > s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25))))
    
    outliers_found = full_df[full_df['is_outlier']]
    
    if not outliers_found.empty:
        print("Outliers encontrados:")
        # <<< MUDANÇA 2: A coluna 'run' foi removida do print >>>
        print(outliers_found[['scenario', 'mode', 'model_id', 'duration_seconds']])
    else:
        print("Nenhum outlier significativo encontrado.")
        
    scenarios = sorted(full_df['scenario'].unique())
    print(f"\nCenários detectados: {scenarios}")
    
    all_scenario_reports = []
    for scenario in scenarios:
        print(f"\n--- Analisando Estatísticas para o Cenário: {scenario} ---")
        
        stats_off = analyze_results(full_df, 'OFF', scenario)
        stats_on = analyze_results(full_df, 'ON', scenario)
        
        if stats_off is None or stats_on is None:
            continue

        final_report = pd.merge(stats_off, stats_on, on='model_id', how='outer')
        
        metrics_for_overhead = ['duration_seconds', 'energy_consumed_kWh', 'gpu_energy_kWh', 'cpu_energy_kWh']
        for metric in metrics_for_overhead:
            mean_off = f'{metric}_mean_OFF'
            mean_on = f'{metric}_mean_ON'
            overhead_col = f'{metric}_overhead_%'
            
            if mean_off in final_report.columns and mean_on in final_report.columns:
                final_report[overhead_col] = ((final_report[mean_on] - final_report[mean_off]) / final_report[mean_off]) * 100
        
        final_report['scenario'] = scenario
        all_scenario_reports.append(final_report)

    if all_scenario_reports:
        full_final_report = pd.concat(all_scenario_reports).reset_index()
        
        cols = full_final_report.columns.tolist()
        cols.insert(0, cols.pop(cols.index('scenario')))
        cols.insert(1, cols.pop(cols.index('model_id')))
        full_final_report = full_final_report.loc[:, cols]
        
        output_filename = 'final_summary_report_with_outliers_removed.csv'
        full_final_report.to_csv(output_filename, index=False)
        
        print(f"\nAnálise completa concluída! Relatório final salvo em: {output_filename}")
        print("\n--- Amostra do Relatório Final ---")
        print(full_final_report[['scenario', 'model_id', 'duration_seconds_mean_OFF', 'duration_seconds_mean_ON', 'duration_seconds_overhead_%']].head())
    else:
        print("\nNenhuma análise pôde ser concluída.")

if __name__ == '__main__':
    main()
