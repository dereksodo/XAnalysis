#!/usr/bin/env bash
# run_all.sh – one-click run for all analyses
# -------------------------------------------
set -e                                  # 任何步骤报错即中断
SECONDS=0                               # 计时器

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== Running pipeline_normal ==="
python3 -m src.Chapter4.pipeline_normal \
        2>&1 | tee "$LOG_DIR/pipeline_normal.log"

echo "=== Running pipeline_time_tuning ==="
python3 -m src.Chapter4.pipeline_time_tuning \
        2>&1 | tee "$LOG_DIR/pipeline_time_tuning.log"

echo "=== Generating plots (plot) ==="
python3 -m src.Chapter4.plot \
        2>&1 | tee "$LOG_DIR/plot.log"

echo "=== Generating plots (plot_tuning) ==="
python3 -m src.Chapter4.plot_tuning \
        2>&1 | tee "$LOG_DIR/plot_tuning.log"

echo "=== Generating plots (figure1) ==="
python3 -m src.Chapter4.figure1 \
        2>&1 | tee "$LOG_DIR/figure1.log"

echo "=== Generating plots (figure2) ==="
python3 -m src.Chapter4.figure2 \
        2>&1 | tee "$LOG_DIR/figure2.log"

echo "=== Generating plots (pipeline_time) ==="
python3 -m src.Chapter5.pipeline_time \
        2>&1 | tee "$LOG_DIR/pipeline_time.log"

echo "=== Generating plots (pipeline_time_tuning) ==="
python3 -m src.Chapter5.pipeline_time_tuning \
        2>&1 | tee "$LOG_DIR/pipeline_time_tuning.log"

echo "=== Generating plots (plot_time) ==="
python3 -m src.Chapter5.plot_time \
        2>&1 | tee "$LOG_DIR/plot_time.log"

echo "=== Generating plots (plot_time_2) ==="
python3 -m src.Chapter5.plot_time_2 \
        2>&1 | tee "$LOG_DIR/plot_time_2.log"


DURATION=$SECONDS
echo "-------------------------------------------"
echo "All tasks completed in $((DURATION/60)) min $((DURATION%60)) sec"
echo "Logs saved to   $LOG_DIR/*.log"