/* W B Langdon at MUN 10 May 2007
 * Program to demonstarte use of OpenGL's glDrawPixels
 */

#include "common.hxx"

#ifdef STD_THREADING
# include <thread>
# ifndef _WIN32
#  include <unistd.h>
# endif
#endif

void cleanup()
{
    // nothing extra to cleanup
}

/* Updates at each timestep */
void thread_update(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax)
{
    int xsize = x_cells - 1;
    int ysize = y_cells - 1;
    int zsize = z_cells - 1;
    
    int temp = 0, position = 0;
    float sum;

    for (int i = xmin; i < xmax; ++i) {
        for (int j = ymin; j < ymax; ++j) {
            for (int k = zmin; k < zmax; ++k) {
                position = XYZINDEX(i, j, k, x_cells, y_cells);
                sum = 0.0f;

                // just the sides 
                sum += heat_matrix[XYZINDEX(i, j, k, x_cells, y_cells)];

                if (i > 0)
                    sum += heat_matrix[XYZINDEX(i-1, j, k, x_cells, y_cells)];
                if (i < xsize)
                    sum += heat_matrix[XYZINDEX(i+1, j, k, x_cells, y_cells)];

                if (j > 0)
                    sum += heat_matrix[XYZINDEX(i, j-1, k, x_cells, y_cells)];
                if (j < ysize)
                    sum += heat_matrix[XYZINDEX(i, j+1, k, x_cells, y_cells)];

                if (k > 0)
                    sum += heat_matrix[XYZINDEX(i, j, k-1, x_cells, y_cells)];
                if (k < zsize)
                    sum += heat_matrix[XYZINDEX(i, j, k+1, x_cells, y_cells)];

                next_heat_matrix[position] = sum / 7.0f;

                if (next_heat_matrix[position] <= 0.001f)
                    next_heat_matrix[position] = 0.0f;
            }
        }
    }
}

void update()
{
#ifndef STD_THREADING
    thread_update(0, x_cells, 0, y_cells, 0, z_cells);
#else
# ifndef _WIN32
    int numthreads = sysconf(_SC_NPROCESSORS_ONLN) - 1;
#  else
    int numthreads = std::threads::hardware_concurrency() - 1;
# endif
    int xwidth = x_cells / (numthreads + 1);

    int xmin = 0;
    int xmax;

    // create our threads
    std::thread *threads = new std::thread[numthreads];
    for (int i = 0; i < numthreads; ++i) {
        xmax = xmin + xwidth;

        threads[i] = std::thread(thread_update, xmin, xmax, 0, y_cells, 0, z_cells);

        xmin = xmax;
    }

    thread_update(xmin, x_cells, 0, y_cells, 0, z_cells);

    // join our threads
    for (int i = 0; i < numthreads; ++i)
        threads[i].join();

    // delete our threads
    delete[] threads;
#endif // STD_THREADING
}

int main(int argc, char** argv)
{
    // initialize our data
    if (!initialize(argc, argv, &update, &cleanup)) return 1;

    // start up opengl
    startOpengl(argc, argv);
}
