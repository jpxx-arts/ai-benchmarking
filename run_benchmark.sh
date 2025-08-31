#!/bin/bash

# --- CONFIGURAÇÕES GERAIS ---
TOTAL_RUNS=30

# --- VALIDAÇÃO DO INPUT ---
if [ "$1" != "on" ] && [ "$1" != "off" ]; then
  echo "Erro: Forneça um argumento ('on' ou 'off') para definir o CC Mode."
  echo "Uso: ./run_experiment.sh on"
  echo "  ou: ./run_experiment.sh off"
  exit 1
fi

# Define o modo CC com base no argumento
if [ "$1" == "on" ]; then
  EXEC_MODE="CC_MODE_ON"
else
  EXEC_MODE="CC_MODE_OFF"
fi

echo "========================================================"
echo "    INICIANDO EXPERIMENTO COMPLETO - MODO: $EXEC_MODE"
echo "========================================================"

# Define os cenários de custo computacional que queremos rodar
COST_SCENARIOS=("low_cost" "high_cost")

# --- LOOP SOBRE OS CENÁRIOS (BAIXO E ALTO CUSTO) ---
for scenario in "${COST_SCENARIOS[@]}"; do

  echo ""
  echo "--------------------------------------------------------"
  echo "    INICIANDO CENÁRIO: $scenario"
  echo "--------------------------------------------------------"

  # Define um arquivo de progresso único para cada cenário e modo
  PROGRESS_FILE="progress_${scenario}_${EXEC_MODE}.log"

  # Lógica de resumo para este cenário específico
  if [ -f "$PROGRESS_FILE" ]; then
    LAST_COMPLETED_RUN=$(cat "$PROGRESS_FILE")
    START_RUN=$(($LAST_COMPLETED_RUN + 1))
    echo "Arquivo de progresso ($PROGRESS_FILE) encontrado. Retomando da execução #$START_RUN."
  else
    START_RUN=1
    echo "Nenhum arquivo de progresso encontrado. Iniciando da execução #1."
  fi

  if [ "$START_RUN" -gt "$TOTAL_RUNS" ]; then
    echo "Cenário '$scenario' já concluído com $TOTAL_RUNS execuções. Pulando."
    continue # Pula para o próximo cenário no loop
  fi

  # Loop principal para as execuções deste cenário
  for i in $(seq $START_RUN $TOTAL_RUNS); do
    echo ""
    echo "--- [$(date)] Iniciando Execução #$i de $TOTAL_RUNS (Cenário: $scenario) ---"

    # Modifica o modo no arquivo main.py
    sed -i "s/EXECUTION_MODE = '.*'/EXECUTION_MODE = '$EXEC_MODE'/" main.py

    # Executa o script Python, passando o cenário como argumento
    python main.py "$scenario" &&
      echo "$i" >"$PROGRESS_FILE"

    if [ $? -eq 0 ]; then
      echo "--- [$(date)] Execução #$i concluída. Progresso salvo em $PROGRESS_FILE. ---"
    else
      echo "--- [$(date)] ERRO: Execução #$i falhou. O progresso NÃO foi salvo. ---"
      echo "Saindo do script. Na próxima execução, ele tentará a execução #$i novamente."
      exit 1
    fi
  done
done

echo ""
echo "========================================================"
echo "    TODOS OS CENÁRIOS CONCLUÍDOS PARA O MODO $EXEC_MODE"
echo "========================================================"
