# proyecto-paralelismo
Implementacion algoritmos basados en stencil, secuenciales y paralelos.
---
Repositorio con implementaciones paralelas de simulaciones de difusión de calor utilizando diferentes stencils y modelos de ejecución:

- **Stencil de 5 puntos (2D)**
- **Stencil de 7 puntos (3D)**
- **Stencil de 9 puntos (2D)**

Cada implementación contiene versiones:

- Secuencial (CPU)
- OpenMP (CPU paralelo)
- CUDA memoria global (GPU)
- CUDA memoria compartida (GPU optimizada)

Además, cada carpeta incluye scripts para ejecutar experimentos de rendimiento y generar archivos CSV con los resultados.

---
# Requisitos

## Hardware

Para ejecutar las versiones CUDA se requiere una GPU NVIDIA.

Los experimentos fueron configurados para:

```
GPU: NVIDIA RTX 2080
CUDA Architecture: sm_75
```

Si se utiliza otra GPU, modificar:

```bash
ARCH="sm_75"
```

en los scripts `.sh`.

---

## Software

Se requiere:

- GCC/G++
- OpenMP
- CUDA Toolkit
- NVIDIA Driver

Verificar CUDA:

```bash
nvcc --version
```

Verificar compilador:

```bash
g++ --version
```

---

# Ejecución de experimentos

Cada stencil posee su propio script de ejecución rápida.

Los scripts realizan automáticamente:

1. Compilación de todas las versiones.
2. Ejecución de experimentos.
3. Medición de tiempo.
4. Cálculo de MLUPS.
5. Generación de archivos CSV.

---

# Stencil 5 puntos

Entrar a la carpeta:

```bash
cd 5p
```

Dar permisos de ejecución:

```bash
chmod +x run_experiments_quick.sh
```

Ejecutar:

```bash
./run_experiments_quick.sh
```

---

# Stencil 7 puntos

Entrar a la carpeta:

```bash
cd 7p
```

Dar permisos:

```bash
chmod +x run_experiments_quick_s7p.sh
```

Ejecutar:

```bash
./run_experiments_quick_s7p.sh
```

---

# Stencil 9 puntos

Entrar a la carpeta:

```bash
cd 9p
```

Dar permisos:

```bash
chmod +x run_experiments_quick_s9p.sh
```

Ejecutar:

```bash
./run_experiments_quick_s9p.sh
```

---

# Configuración de experimentos

Los scripts utilizan una versión reducida de experimentos para mantener un tiempo de ejecución aproximado de 30 minutos.

## Stencils 5p y 9p

Tamaños de grilla:

```
512
1024
2048
```

Además, CUDA ejecuta una prueba adicional:

```
4096
```

solo para:

- cuda_global
- cuda_shared


---

## Stencil 7p

Tamaños de grilla:

```
64
128
256
```

CUDA ejecuta adicionalmente:

```
512
```

solo para las versiones GPU.

---

# Repeticiones

Cada configuración se ejecuta:

```
3 veces
```

para obtener mediciones más estables.

---

# OpenMP

Los experimentos OpenMP utilizan:

```
1
4
8
12
```

hilos.

La cantidad se controla mediante:

```bash
OMP_NUM_THREADS
```

Ejemplo:

```bash
OMP_NUM_THREADS=8 ./bin/omp_1024
```

---

# Resultados

Los resultados se almacenan automáticamente en la carpeta:

```
results/
```

Cada script genera un archivo CSV.

Ejemplos:

Stencil 5 puntos:

```
results/resultados_quick.csv
```

Stencil 7 puntos:

```
results/resultados_quick_s7p.csv
```

Stencil 9 puntos:

```
results/resultados_quick_s9p.csv
```

---

# Formato CSV

Cada archivo contiene:

| Campo | Descripción |
|-|-|
| algoritmo | versión ejecutada |
| grid_size | tamaño de la grilla |
| hilos | cantidad de hilos OpenMP o GPU |
| repeticion | número de repetición |
| sample_id | muestra utilizada |
| tiempo_s | tiempo de ejecución en segundos |
| mlups | millones de actualizaciones por segundo |
| checksum | verificación del resultado |
