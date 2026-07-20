#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;

// =====================================================
// CONFIGURACION
// =====================================================

// Se puede sobreescribir en compilacion con -DGRID_SIZE=N -DSTEPS=N
#ifndef GRID_SIZE
#define GRID_SIZE 2048
#endif

#ifndef STEPS
#define STEPS 1000
#endif

constexpr double alpha = 0.1;
constexpr double dx = 1.0;
constexpr double dt = 0.1;

constexpr double r = alpha * dt / (dx * dx);

const string CSV_FILE = "dataset_400.csv";

// Tamano de bloque CUDA (2D)
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

// =====================================================
// MACRO PARA CHEQUEAR ERRORES CUDA
// =====================================================

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err__ = (call);                                   \
        if (err__ != cudaSuccess) {                                   \
            throw runtime_error(                                      \
                string("CUDA error: ") +                              \
                cudaGetErrorString(err__) +                            \
                " en " + __FILE__ + ":" + to_string(__LINE__));        \
        }                                                              \
    } while (0)

// =====================================================
// CARGA UNA MUESTRA DEL CSV 
// =====================================================

vector<double> load_sample(
    const string& filename,
    int sample_id)
{
    ifstream file(filename);

    if (!file.is_open())
    {
        throw runtime_error(
            "No se pudo abrir el archivo: " +
            filename);
    }

    string line;
    int current_row = 0;

    while (getline(file, line))
    {
        if (current_row == sample_id)
        {
            vector<double> values;

            stringstream ss(line);
            string cell;

            while (getline(ss, cell, ','))
            {
                values.push_back(stod(cell));
            }

            return values;
        }

        current_row++;
    }

    throw runtime_error(
        "sample_id fuera de rango");
}

// =====================================================
// GENERAR CONDICION INICIAL DESDE LOS 53 PARAMETROS
// =====================================================

void seed_from_dataset(
    vector<double>& grid,
    int N,
    const vector<double>& sample)
{
    const int n_hotspots =
        static_cast<int>(sample.size());

    for (int k = 0; k < n_hotspots; k++)
    {
        double t =
            double(k) /
            double(n_hotspots - 1);

        int x =
            int(0.1 * N + 0.8 * N * t);

        int y =
            int(
                0.5 * N +
                0.35 * N *
                sin(2.0 * M_PI * t)
            );

        double temp =
            sample[k] * 100.0;

        grid[
            static_cast<size_t>(x) * N + y
        ] = temp;
    }
}

// =====================================================
// KERNEL CUDA - MEMORIA GLOBAL (stencil de 9 puntos)
// =====================================================

__global__ void heat_kernel_global(
    const double* __restrict__ u,
    double* __restrict__ u_new,
    int N,
    double r)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // columna
    int i = blockIdx.y * blockDim.y + threadIdx.y; // fila

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1)
    {
        size_t idx = static_cast<size_t>(i) * N + j;

        double cardinales =
            u[idx + N]
            +
            u[idx - N]
            +
            u[idx + 1]
            +
            u[idx - 1];

        double diagonales =
            u[idx + N + 1]
            +
            u[idx + N - 1]
            +
            u[idx - N + 1]
            +
            u[idx - N - 1];

        double laplaciano =
            (
                4.0 * cardinales
                +
                diagonales
                -
                20.0 * u[idx]
            ) / 6.0;

        u_new[idx] =
            u[idx]
            +
            r * laplaciano;
    }
}

// =====================================================
// MAIN
// =====================================================

int main(int argc, char* argv[])
{
    try
    {
        int sample_id = 0;

        if (argc > 1)
        {
            sample_id = stoi(argv[1]);
        }

        cout << "=====================================\n";
        cout << "2D Heat Diffusion Benchmark (9-point, CUDA - Global Memory)\n";
        cout << "=====================================\n";

        cout << "Grid Size : "
             << GRID_SIZE << "x"
             << GRID_SIZE << '\n';

        cout << "Steps     : "
             << STEPS << '\n';

        cout << "Sample ID : "
             << sample_id << '\n';

        cout << "CSV       : "
             << CSV_FILE << "\n\n";

        // ----------------------------------
        // Cargar muestra
        // ----------------------------------

        vector<double> sample =
            load_sample(
                CSV_FILE,
                sample_id);

        if (sample.size() != 53)
        {
            cerr
                << "ERROR: se esperaban 53 parametros y se cargaron "
                << sample.size()
                << endl;

            return 1;
        }

        cout << "Parametros cargados correctamente: "
             << sample.size()
             << "\n\n";

        // ----------------------------------
        // Reservar memoria host
        // ----------------------------------

        size_t cells =
            static_cast<size_t>(GRID_SIZE)
            * GRID_SIZE;

        size_t bytes = cells * sizeof(double);

        vector<double> u(cells, 0.0);

        // ----------------------------------
        // Sembrar condicion inicial (en host)
        // ----------------------------------

        seed_from_dataset(
            u,
            GRID_SIZE,
            sample);

        // ----------------------------------
        // Reservar memoria device
        // ----------------------------------

        double* d_u = nullptr;
        double* d_u_new = nullptr;

        CUDA_CHECK(cudaMalloc(&d_u, bytes));
        CUDA_CHECK(cudaMalloc(&d_u_new, bytes));

        CUDA_CHECK(cudaMemcpy(
            d_u, u.data(), bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_u_new, 0, bytes));

        // ----------------------------------
        // Configuracion de lanzamiento
        // ----------------------------------

        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid(
            (GRID_SIZE + BLOCK_X - 1) / BLOCK_X,
            (GRID_SIZE + BLOCK_Y - 1) / BLOCK_Y);

        // ----------------------------------
        // Benchmark
        // ----------------------------------

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        for (int step = 0; step < STEPS; step++)
        {
            heat_kernel_global<<<grid, block>>>(
                d_u, d_u_new, GRID_SIZE, r);

            swap(d_u, d_u_new);
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double elapsed = elapsed_ms / 1000.0;

        // ----------------------------------
        // Copiar resultado a host
        // ----------------------------------

        CUDA_CHECK(cudaMemcpy(
            u.data(), d_u, bytes, cudaMemcpyDeviceToHost));

        // ----------------------------------
        // MLUPS
        // ----------------------------------

        double mlups =
            (
                static_cast<double>(GRID_SIZE)
                * GRID_SIZE
                * STEPS
            )
            /
            (
                elapsed * 1e6
            );

        // ----------------------------------
        // Checksum
        // ----------------------------------

        double checksum = 0.0;

        for (double v : u)
        {
            checksum += v;
        }

        // ----------------------------------
        // Resultados
        // ----------------------------------

        cout << "=====================================\n";
        cout << "Tiempo   : "
             << elapsed
             << " s\n";

        cout << "MLUPS    : "
             << mlups
             << '\n';

        cout << "Checksum : "
             << checksum
             << '\n';

        cout << "=====================================\n";

        // ----------------------------------
        // Liberar memoria
        // ----------------------------------

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_u);
        cudaFree(d_u_new);
    }
    catch (const exception& e)
    {
        cerr << "\nERROR: "
             << e.what()
             << endl;

        return 1;
    }

    return 0;
}

// =====================================================
// COMPILACION
// nvcc -O3 -arch=sm_75 s9p_global.cu -o s9p_global
// (ajustar -arch a la arquitectura de la GPU disponible)
// =====================================================
