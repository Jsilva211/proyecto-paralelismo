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

// 64
// 128
// 256
// 512

// Se puede sobreescribir en compilacion con -DGRID_SIZE=N -DSTEPS=N
#ifndef GRID_SIZE
#define GRID_SIZE 256
#endif

#ifndef STEPS
#define STEPS 600
#endif

constexpr double alpha = 0.1;
constexpr double dx = 1.0;
constexpr double dt = 0.1;

constexpr double r = alpha * dt / (dx * dx);

const string CSV_FILE = "dataset_400.csv";

// Tamano del tile Y-Z (sin halo). El bloque CUDA tiene el mismo tamano.
// La dimension X NO se tilea: cada bloque recorre X completo con un
// bucle interno (plane-sweep), reutilizando el mismo tile de memoria
// compartida para cada plano.
constexpr int TILE_Y = 16;
constexpr int TILE_Z = 16;

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
// INDEXADOR 3D -> 1D (host)
// =====================================================

inline size_t index3D_host(
    int x,
    int y,
    int z)
{
    return
        static_cast<size_t>(x)
        * GRID_SIZE
        * GRID_SIZE
        +
        static_cast<size_t>(y)
        * GRID_SIZE
        +
        z;
}

// =====================================================
// GENERAR CONDICION INICIAL DESDE LOS 53 PARAMETROS
// =====================================================

void seed_from_dataset(
    vector<double>& grid,
    const vector<double>& sample)
{
    const int n_hotspots =
        static_cast<int>(sample.size());

    for (int k = 0;
         k < n_hotspots;
         k++)
    {
        double t =
            double(k)
            /
            double(n_hotspots - 1);

        // Posición X avanza
        int x =
            int(
                0.10 * GRID_SIZE +
                0.80 * GRID_SIZE * t
            );

        // Hélice 3D
        int y =
            int(
                0.50 * GRID_SIZE +
                0.30 * GRID_SIZE *
                sin(4.0 * M_PI * t)
            );

        int z =
            int(
                0.50 * GRID_SIZE +
                0.30 * GRID_SIZE *
                cos(4.0 * M_PI * t)
            );

        x = max(1, min(GRID_SIZE - 2, x));
        y = max(1, min(GRID_SIZE - 2, y));
        z = max(1, min(GRID_SIZE - 2, z));

        double temp =
            sample[k] * 100.0;

        grid[
            index3D_host(x, y, z)
        ] = temp;
    }
}

// =====================================================
// KERNEL CUDA - MEMORIA COMPARTIDA (stencil 3D de 7 puntos)
// Tecnica de "plane-sweep": cada bloque cubre un tile fijo del
// plano Y-Z y recorre toda la dimension X con un bucle interno.
// En cada iteracion se carga en memoria compartida solo el plano
// actual (con halo de 1 en Y y en Z). Los vecinos en X (x-1, x+1)
// no se tilean: se leen directo de memoria global, ya que cada
// hilo los necesita una sola vez por plano y no se reutilizan
// entre hilos vecinos del mismo bloque.
// =====================================================

__global__ void heat_kernel_shared_planesweep(
    const double* __restrict__ u,
    double* __restrict__ u_new,
    int N,
    double r)
{
    __shared__ double tile[TILE_Y + 2][TILE_Z + 2];

    int tz = threadIdx.x;
    int ty = threadIdx.y;

    int z = blockIdx.x * TILE_Z + tz;
    int y = blockIdx.y * TILE_Y + ty;

    int lz = tz + 1;
    int ly = ty + 1;

    // Helper para leer con proteccion de bordes globales
    auto safe_read = [&](int gx, int gy, int gz) -> double
    {
        return (gx >= 0 && gx < N && gy >= 0 && gy < N && gz >= 0 && gz < N)
            ? u[
                static_cast<size_t>(gx) * N * N
                +
                static_cast<size_t>(gy) * N
                +
                gz
              ]
            : 0.0;
    };

    for (int x = 1; x < N - 1; x++)
    {
        // Cargar el plano Y-Z actual en memoria compartida
        tile[ly][lz] = safe_read(x, y, z);

        // Halo izquierdo / derecho en Z
        if (tz == 0)
        {
            tile[ly][0] = safe_read(x, y, z - 1);
        }

        if (tz == TILE_Z - 1)
        {
            tile[ly][TILE_Z + 1] = safe_read(x, y, z + 1);
        }

        // Halo superior / inferior en Y
        if (ty == 0)
        {
            tile[0][lz] = safe_read(x, y - 1, z);
        }

        if (ty == TILE_Y - 1)
        {
            tile[TILE_Y + 1][lz] = safe_read(x, y + 1, z);
        }

        __syncthreads();

        if (y >= 1 && y < N - 1 && z >= 1 && z < N - 1)
        {
            size_t idx =
                static_cast<size_t>(x) * N * N
                +
                static_cast<size_t>(y) * N
                +
                z;

            // Vecinos en X: directo de memoria global (sin tiling)
            double u_xm1 = u[idx - static_cast<size_t>(N) * N];
            double u_xp1 = u[idx + static_cast<size_t>(N) * N];

            u_new[idx] =
                tile[ly][lz]
                +
                r *
                (
                    u_xp1
                    +
                    u_xm1

                    +
                    tile[ly + 1][lz]
                    +
                    tile[ly - 1][lz]

                    +
                    tile[ly][lz + 1]
                    +
                    tile[ly][lz - 1]

                    -
                    6.0 * tile[ly][lz]
                );
        }

        // Necesario antes de sobreescribir el tile en la proxima
        // iteracion del plane-sweep (siguiente plano X).
        __syncthreads();
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
        cout << "3D Heat Diffusion Benchmark (CUDA - Shared Memory, plane-sweep)\n";
        cout << "=====================================\n";

        cout << "Grid Size : "
             << GRID_SIZE << "x"
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
            * GRID_SIZE
            * GRID_SIZE;

        size_t bytes = cells * sizeof(double);

        vector<double> u(cells, 0.0);

        // ----------------------------------
        // Sembrar condicion inicial (en host)
        // ----------------------------------

        seed_from_dataset(
            u,
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
        // Grid 2D: cada bloque cubre un tile del plano Y-Z y
        // recorre toda la dimension X internamente.
        // ----------------------------------

        dim3 block(TILE_Z, TILE_Y);
        dim3 grid(
            (GRID_SIZE + TILE_Z - 1) / TILE_Z,
            (GRID_SIZE + TILE_Y - 1) / TILE_Y);

        // ----------------------------------
        // Benchmark
        // ----------------------------------

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        for (int step = 0; step < STEPS; step++)
        {
            heat_kernel_shared_planesweep<<<grid, block>>>(
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
                static_cast<double>(cells)
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
// nvcc -O3 -arch=sm_75 s7p_shared.cu -o s7p_shared
// (ajustar -arch a la arquitectura de la GPU disponible)
// =====================================================
