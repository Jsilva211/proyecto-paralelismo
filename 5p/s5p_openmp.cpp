#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <omp.h>

using namespace std;

// =====================================================
// CONFIGURACION
// =====================================================

// 512
// 1024
// 2048
// 4096

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

        // El numero de hilos lo controla la variable de entorno
        // OMP_NUM_THREADS (asi el script puede barrerla sin recompilar).
        // Si no esta seteada, OpenMP usa su default (normalmente el
        // numero de hilos logicos disponibles).

        cout << "=====================================\n";
        cout << "Heat Diffusion Benchmark (OpenMP - sin collapse)\n";
        cout << "=====================================\n";

        cout << "Grid Size : "
             << GRID_SIZE << "x"
             << GRID_SIZE << '\n';

        cout << "Steps     : "
             << STEPS << '\n';

        cout << "Threads   : "
             << omp_get_max_threads() << '\n';

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

        cout
            << "Parametros cargados correctamente: "
            << sample.size()
            << "\n\n";

        // ----------------------------------
        // Reservar memoria
        // ----------------------------------

        size_t cells =
            static_cast<size_t>(GRID_SIZE)
            * GRID_SIZE;

        vector<double> u(cells, 0.0);
        vector<double> u_new(cells, 0.0);

        // ----------------------------------
        // Sembrar condicion inicial
        // ----------------------------------

        seed_from_dataset(
            u,
            GRID_SIZE,
            sample);

        // ----------------------------------
        // Benchmark
        // ----------------------------------

        auto start =
            chrono::high_resolution_clock::now();

        for (int step = 0;
             step < STEPS;
             step++)
        {
            // Sin collapse(2): solo se paraleliza el bucle externo i.
            // El bucle interno j queda intacto para que el compilador
            // pueda auto-vectorizarlo igual que en la version secuencial.
            #pragma omp parallel for schedule(static)
            for (int i = 1;
                 i < GRID_SIZE - 1;
                 i++)
            {
                for (int j = 1;
                     j < GRID_SIZE - 1;
                     j++)
                {
                    size_t idx =
                        static_cast<size_t>(i)
                        * GRID_SIZE
                        + j;

                    u_new[idx] =
                        u[idx]
                        +
                        r *
                        (
                            u[idx + GRID_SIZE]
                            +
                            u[idx - GRID_SIZE]
                            +
                            u[idx + 1]
                            +
                            u[idx - 1]
                            -
                            4.0 * u[idx]
                        );
                }
            }

            swap(u, u_new);
        }

        auto end =
            chrono::high_resolution_clock::now();

        double elapsed =
            chrono::duration<double>(
                end - start
            ).count();

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
        // Checksum (SECUENCIAL)
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