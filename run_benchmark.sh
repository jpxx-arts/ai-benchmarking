#!/bin/bash

# --- CONFIGURAÇÕES ---
EXEC_MODE="CC_MODE_OFF" 
TOTAL_RUNS=30
PROGRESS_FILE="progress.log"

# --- LÓGICA DE RESUMO ---
# Verifica qual foi a última execução CONCLUÍDA com sucesso.
if [ -f "$PROGRESS_FILE" ]; then
    LAST_COMPLETED_RUN=$(cat "$PROGRESS_FILE")
    START_RUN=$(($LAST_COMPLETED_RUN + 1))
    echo "Arquivo de progresso encontrado. Última execução concluída foi a #$LAST_COMPLETED_RUN."
    echo "Retomando a partir da execução #$START_RUN."
else
    LAST_COMPLETED_RUN=0
    START_RUN=1
    echo "Nenhum arquivo de progresso encontrado. Iniciando da execução #1."
fi

if [ "$START_RUN" -gt "$TOTAL_RUNS" ]; then
    echo "Benchmark já concluído com $TOTAL_RUNS execuções. Nada a fazer."
    exit 0
fi


# --- LOOP PRINCIPAL DE EXECUÇÃO ---
echo "Iniciando benchmark para o modo $EXEC_MODE (Execuções de $START_RUN a $TOTAL_RUNS)..."

for i in $(seq $START_RUN $TOTAL_RUNS)
do
  echo "--- [$(date)] Iniciando Execução #$i de $TOTAL_RUNS ---"
  
  sed -i "s/EXECUTION_MODE = '.*'/EXECUTION_MODE = '$EXEC_MODE'/" main.py
  
  python main.py && \
  echo "$i" > "$PROGRESS_FILE"
  
  if [ $? -eq 0 ]; then
      echo "--- [$(date)] Execução #$i concluída com sucesso. Progresso salvo. ---"
  else
      echo "--- [$(date)] ERRO: A execução #$i falhou. O progresso NÃO foi salvo. ---"
      echo "Saindo do script. Na próxima execução, ele tentará a execução #$i novamente."
      exit 1
  fi
done

echo "Benchmark concluído com sucesso para o modo $EXEC_MODE."
