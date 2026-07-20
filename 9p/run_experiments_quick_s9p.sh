#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# CONFIGURACION DEL EXPERIMENTO - s9p (2D, stencil 9 puntos)
# VERSION RAPIDA, ~30 min
#
# Misma logica que run_experiments_quick.sh (s5p), aplicada a
# s9p. Al ser 2D igual que s5p, se usan los mismos tamanos de
# grilla: el costo por celda es mayor (9 vecinos vs 5), pero
# escala de la misma forma con N.
#
#   - GRID_SIZES (secuencial+omp+cuda): 512/1024/2048
#     (se quito 4096 de aqui porque en CPU/OpenMP es el que
#      mas tiempo consume: 4x el costo de 2048 por repeticion)
#   - CUDA_EXTRA_GRID_SIZES: 4096 SOLO para las versiones CUDA
#     (en GPU esa grilla sigue siendo barata, segundos por
#      corrida, asi que se agrega como punto extra de escala
#      sin afectar el presupuesto de tiempo)
#   - REPETITIONS: 3
#   - THREAD_COUNTS: 4 valores (1,4,8,12)
#
# =====================================================

GRID_SIZES=(512 1024 2048)
CUDA_EXTRA_GRID_SIZES=(4096)     # solo para cuda_global / cuda_shared
THREAD_COUNTS=(1 4 8 12)
REPETITIONS=3
SAMPLE_ID=0
ARCH="sm_75"                    # RTX 2080 (Turing)

SRC_DIR="."
BIN_DIR="./bin_s9p"
RESULTS_DIR="./results"
RESULTS_CSV="$RESULTS_DIR/resultados_quick_s9p.csv"

SEQ_SRC="$SRC_DIR/s9p.cpp"
OMP_SRC="$SRC_DIR/s9p_openmp.cpp"     # debe existir en el mismo directorio
GLOBAL_SRC="$SRC_DIR/s9p_global.cu"
SHARED_SRC="$SRC_DIR/s9p_shared.cu"

mkdir -p "$BIN_DIR" "$RESULTS_DIR"

# =====================================================
# FASE 1: COMPILACION
# =====================================================

echo "========================================="
echo " FASE 1: COMPILACION (s9p, version rapida)"
echo "========================================="

for f in "$SEQ_SRC" "$GLOBAL_SRC" "$SHARED_SRC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: no se encontro $f"
        exit 1
    fi
done

HAS_OMP=0
if [[ -f "$OMP_SRC" ]]; then
    HAS_OMP=1
else
    echo "AVISO: no se encontro $OMP_SRC -> se omitira la version OpenMP."
    echo "       (agregale el bloque #ifndef GRID_SIZE / #define GRID_SIZE 2048"
    echo "        / #endif antes del constexpr para poder barrer tamanos igual"
    echo "        que en las otras versiones)"
fi

ALL_GRID_SIZES=("${GRID_SIZES[@]}" "${CUDA_EXTRA_GRID_SIZES[@]}")

for N in "${GRID_SIZES[@]}"; do
    echo "--- GRID_SIZE=$N (secuencial + omp + cuda) ---"

    echo "  [cpu]  compilando secuencial..."
    g++ -O3 -march=native -DGRID_SIZE="$N" -o "$BIN_DIR/seq_$N" "$SEQ_SRC"

    if [[ "$HAS_OMP" -eq 1 ]]; then
        echo "  [omp]  compilando OpenMP..."
        g++ -O3 -march=native -fopenmp -DGRID_SIZE="$N" -o "$BIN_DIR/omp_$N" "$OMP_SRC"
    fi

    echo "  [cuda] compilando memoria global..."
    nvcc -O3 -arch="$ARCH" -DGRID_SIZE="$N" -o "$BIN_DIR/global_$N" "$GLOBAL_SRC"

    echo "  [cuda] compilando memoria compartida..."
    nvcc -O3 -arch="$ARCH" -DGRID_SIZE="$N" -o "$BIN_DIR/shared_$N" "$SHARED_SRC"
done

for N in "${CUDA_EXTRA_GRID_SIZES[@]}"; do
    echo "--- GRID_SIZE=$N (solo cuda) ---"

    echo "  [cuda] compilando memoria global..."
    nvcc -O3 -arch="$ARCH" -DGRID_SIZE="$N" -o "$BIN_DIR/global_$N" "$GLOBAL_SRC"

    echo "  [cuda] compilando memoria compartida..."
    nvcc -O3 -arch="$ARCH" -DGRID_SIZE="$N" -o "$BIN_DIR/shared_$N" "$SHARED_SRC"
done

echo
echo "Compilacion completa. Binarios en $BIN_DIR/"
echo

# =====================================================
# FASE 2: EJECUCION DE EXPERIMENTOS
# =====================================================

echo "========================================="
echo " FASE 2: EJECUCION (s9p, version rapida)"
echo "========================================="

echo "algoritmo,grid_size,hilos,repeticion,sample_id,tiempo_s,mlups,checksum" > "$RESULTS_CSV"

# Uso: run_and_parse <algoritmo> <grid_size> <hilos> <rep> <binario> [VAR=val ...]
run_and_parse() {
    local algoritmo="$1"
    local grid_size="$2"
    local hilos="$3"
    local rep="$4"
    local binpath="$5"
    shift 5
    local extra_env=("$@")

    echo "  -> $algoritmo | grid=$grid_size | hilos=$hilos | rep=$rep"

    local output
    if ! output=$(env "${extra_env[@]}" "$binpath" "$SAMPLE_ID" 2>&1); then
        echo "     ERROR ejecutando $binpath"
        echo "$algoritmo,$grid_size,$hilos,$rep,$SAMPLE_ID,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
        return
    fi

    local tiempo mlups checksum
    tiempo=$(echo "$output"   | grep "Tiempo"   | awk -F':' '{print $2}' | awk '{print $1}')
    mlups=$(echo "$output"    | grep "MLUPS"    | awk -F':' '{print $2}' | awk '{print $1}')
    checksum=$(echo "$output" | grep "Checksum" | awk -F':' '{print $2}' | awk '{print $1}')

    echo "$algoritmo,$grid_size,$hilos,$rep,$SAMPLE_ID,$tiempo,$mlups,$checksum" >> "$RESULTS_CSV"
}

# ---- Secuencial + OpenMP + CUDA en las grillas chicas/medianas ----
for N in "${GRID_SIZES[@]}"; do
    echo "--- Ejecutando GRID_SIZE=$N ---"

    echo "  (warm-up cuda global / shared, descartado)"
    "$BIN_DIR/global_$N" "$SAMPLE_ID" > /dev/null 2>&1 || true
    "$BIN_DIR/shared_$N" "$SAMPLE_ID" > /dev/null 2>&1 || true

    for rep in $(seq 1 "$REPETITIONS"); do

        run_and_parse "secuencial" "$N" 1 "$rep" "$BIN_DIR/seq_$N"
        run_and_parse "cuda_global" "$N" "GPU" "$rep" "$BIN_DIR/global_$N"
        run_and_parse "cuda_shared" "$N" "GPU" "$rep" "$BIN_DIR/shared_$N"

        if [[ "$HAS_OMP" -eq 1 ]]; then
            for T in "${THREAD_COUNTS[@]}"; do
                run_and_parse "openmp" "$N" "$T" "$rep" "$BIN_DIR/omp_$N" "OMP_NUM_THREADS=$T"
            done
        fi
    done
done

# ---- Solo CUDA en la grilla grande (4096) ----
for N in "${CUDA_EXTRA_GRID_SIZES[@]}"; do
    echo "--- Ejecutando GRID_SIZE=$N (solo cuda) ---"

    echo "  (warm-up cuda global / shared, descartado)"
    "$BIN_DIR/global_$N" "$SAMPLE_ID" > /dev/null 2>&1 || true
    "$BIN_DIR/shared_$N" "$SAMPLE_ID" > /dev/null 2>&1 || true

    for rep in $(seq 1 "$REPETITIONS"); do
        run_and_parse "cuda_global" "$N" "GPU" "$rep" "$BIN_DIR/global_$N"
        run_and_parse "cuda_shared" "$N" "GPU" "$rep" "$BIN_DIR/shared_$N"
    done
done

echo
echo "========================================="
echo " Experimentos completos (s9p)."
echo " Resultados en: $RESULTS_CSV"
echo "========================================="


